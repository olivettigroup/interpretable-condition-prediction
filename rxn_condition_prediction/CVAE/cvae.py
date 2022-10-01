import numpy as np

from keras import backend as K
from keras import metrics
from keras.layers import (GRU, Conv1D, Conv2D, Dense, Dropout, Embedding, Flatten,
                          Input, Lambda, RepeatVector, Reshape,
                          TimeDistributed, Masking)
from keras.layers.merge import Concatenate, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sys, os

class TempTimeGenerator(object):
    def __init__(self):
        self.vae = None
        self.encoder = None
        self.decoder = None
    def build_nn_model(self, 
                    rnn_dim=64, conv_window=3, conv_filters=16,
                    intermediate_dim=128, latent_dim=3, 
                    use_conditionals=True, use_kl_loss=True,
                    temp_time_len=8, target_len=103, precursor_len=515,
                    stdev=1.0, kl_coeff=1.0, rec_coeff=1.0):

        self.latent_dim = latent_dim
        self.original_dim = temp_time_len
        self.temp_time_dim = temp_time_len

        x_temp_time = Input(shape=(temp_time_len, 1), name="temp_time_in")

        conv_x1 = Conv1D(conv_filters, conv_window, padding="valid", activation="relu", name='conv_enc_1')(x_temp_time)
        conv_x2 = Conv1D(conv_filters, conv_window, padding="valid", activation="relu", name='conv_enc_2')(conv_x1)
        # add third layer
        conv_x3 = Conv1D(conv_filters, conv_window, padding="valid", activation="relu", name='conv_enc_3')(conv_x2)
        h_flatten = Flatten()(conv_x3)

        z_mean_func = Dense(latent_dim, name="means_enc")
        z_log_var_func = Dense(latent_dim, name="vars_enc")

        z_mean = z_mean_func(h_flatten)
        z_log_var = z_log_var_func(h_flatten)

        def sample(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(latent_dim,), mean=0.0, stddev=stdev)
            return z_mean + K.exp(z_log_var / 2) * epsilon


        z = Lambda(sample, name="lambda_sample")([z_mean, z_log_var])

        # conditional inputs
        c_material = Input(shape=(target_len,), name="cond_matrl_in")
        c_precursors = Input(shape=(precursor_len,), name="materials_in")


        if use_conditionals:
            z_conditional = Concatenate(name="concat_cond")([z, c_material, c_precursors])
        else:
            z_conditional = z

        decoder_h = Dense(intermediate_dim, activation="relu", name="hidden_dec")
        decoder_h_repeat = RepeatVector(temp_time_len, name="h_rep_dec")
        decoder_h_gru_1 = GRU(rnn_dim, return_sequences=True, name="recurrent_dec_1")
        decoder_h_gru_2 = GRU(rnn_dim, return_sequences=True, name="recurrent_dec_2")
        # add third layer 
        decoder_h_gru_3 = GRU(rnn_dim, return_sequences=True, name="recurrent_dec_3")

        decoder_synth = TimeDistributed(Dense(1), name="synth_decoded")

        h_decoded = decoder_h(z_conditional)
        h_decode_repeat = decoder_h_repeat(h_decoded)
        gru_h_decode_1 = decoder_h_gru_1(h_decode_repeat)
        gru_h_decode_2 = decoder_h_gru_2(gru_h_decode_1)
        gru_h_decode_3 = decoder_h_gru_3(gru_h_decode_2)
        x_decoded_synth = decoder_synth(gru_h_decode_3)

        def vae_xent_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            rec_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            if not use_kl_loss:
                return rec_loss
            return rec_loss + kl_loss

        def vae_mse_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            rec_loss = self.temp_time_dim * metrics.mean_squared_error(x, x_decoded_mean)
            kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            if not use_kl_loss:
                return rec_loss
            return (rec_coeff * rec_loss) + (kl_coeff * kl_loss)

        encoder = Model(inputs=[x_temp_time], outputs=[z_mean, z_log_var, z])

        decoder_x_input = Input(shape=(latent_dim,))

        if use_conditionals:
            decoder_inputs = Concatenate(name="concat_cond_dec")([decoder_x_input, c_material, c_precursors])
        else:
            decoder_inputs = decoder_x_input

        _h_decoded = decoder_h(decoder_inputs)
        _h_decode_repeat = decoder_h_repeat(_h_decoded)
        _gru_h_decode_1 = decoder_h_gru_1(_h_decode_repeat)
        _gru_h_decode_2 = decoder_h_gru_2(_gru_h_decode_1)
        _gru_h_decode_3 = decoder_h_gru_3(_gru_h_decode_2)
        _x_decoded_synth = decoder_synth(_gru_h_decode_3)

        decoder = Model(inputs=[decoder_x_input, c_material, c_precursors], 
                        outputs=[_x_decoded_synth])

        vae = Model(inputs=[x_temp_time, c_material, c_precursors],
                    outputs=[x_decoded_synth])

        vae.compile(
        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=True),
        loss=[vae_mse_loss],
        metrics=['mse']
        )

        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def train(self, inputs, outputs, epochs=300, val_data=None, val_split=0.0, batch_size=16, callbacks=None, verbose=2):
        fitting_results = self.vae.fit(
            x=inputs,
            y=outputs,
            epochs=epochs,
            validation_split=val_split,
            validation_data=val_data,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )

        return fitting_results.history

    def generate_samples(self, target_material, precursors, n_samples=10, random_samples=None, stdev=1.0):
        if random_samples is None:
            random_samples = np.random.normal(scale=stdev, size=(n_samples, self.latent_dim))

        c_material = np.repeat(target_material, repeats=n_samples, axis=0)
        c_precursors = np.repeat(precursors, repeats=n_samples, axis=0)

        return self.decoder.predict([random_samples, c_material, c_precursors])


    def save_models(self, model_variant="default", save_path="/home/synthesis-generation/bin/CJK/paper/"):
        self.vae.save_weights(save_path + model_variant + "_vae.h5")
        self.encoder.save_weights(save_path + model_variant + "_encoder.h5")
        self.decoder.save_weights(save_path + model_variant + "_decoder.h5")

    def load_models(self, model_variant="default", load_path="/home/synthesis-generation/bin/CJK/paper/"):
        self.vae.load_weights(load_path + model_variant + "_vae.h5")
        self.encoder.load_weights(load_path + model_variant + "_encoder.h5")
        self.decoder.load_weights(load_path  + model_variant + "_decoder.h5")