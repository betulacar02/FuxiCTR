import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Layer
from tensorflow.keras.models import Model

class CrossNet(Layer):
    def __init__(self, num_layers, **kwargs):
        super(CrossNet, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.cross_weights = []
        self.cross_bias = []

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        for _ in range(self.num_layers):
            self.cross_weights.append(self.add_weight(shape=(feature_dim, 1), initializer='glorot_uniform', trainable=True))
            self.cross_bias.append(self.add_weight(shape=(feature_dim,), initializer='zeros', trainable=True))

    def call(self, inputs):
        x0 = inputs
        x = x0
        for i in range(self.num_layers):
            x = tf.matmul(x0, tf.matmul(x, self.cross_weights[i])) + self.cross_bias[i] + x
        return x

class DeepFMDCN:
    def __init__(self, feature_sizes, embedding_dim, num_cross_layers):
        self.feature_sizes = feature_sizes
        self.embedding_dim = embedding_dim
        self.num_cross_layers = num_cross_layers

    def build_model(self):
        inputs = []
        embeddings = []

        for feature_size in self.feature_sizes:
            input_layer = Input(shape=(1,), dtype='int32')
            embedding_layer = Embedding(input_dim=feature_size, output_dim=self.embedding_dim, input_length=1)(input_layer)
            inputs.append(input_layer)
            embeddings.append(embedding_layer)

        embeddings_stack = tf.stack(embeddings, axis=1)
        summed_features = tf.reduce_sum(embeddings_stack, axis=1)
        squared_sum = tf.square(summed_features)
        squared_features = tf.reduce_sum(tf.square(embeddings_stack), axis=1)
        fm_interaction = 0.5 * tf.subtract(squared_sum, squared_features)

        dnn_input = Flatten()(tf.concat(embeddings, axis=1))
        dnn = Dense(128, activation='relu')(dnn_input)
        dnn = Dense(64, activation='relu')(dnn)
        dnn = Dense(32, activation='relu')(dnn)

        crossnet_output = CrossNet(self.num_cross_layers)(dnn_input)
        combined_input = Concatenate()([fm_interaction, dnn, crossnet_output])
        output = Dense(1, activation='sigmoid')(combined_input)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
