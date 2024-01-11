from typing import List, Tuple

import tensorflow as tf


class MyNetwork:
    def __init__(
        self,
        First_layer_num_filter: int,
        Kernel_size: int,
        Resnet_version: str,
        Num_residual_blocks: int,
        List_residual_layers: List[int],
    ):
        """Initialize the network with given parameters.

        Args:
            First_layer_num_filter: Number of filters in the first layer.
            Kernel_size: Size of the convolutional kernels.
            Resnet_version: Version of the ResNet ('V1' or 'V2').
            Num_residual_blocks: Number of residual blocks.
            List_residual_layers: List containing the number of layers in each residual block.
        """
        self.First_layer_num_filter = First_layer_num_filter
        self.Kernel_size = Kernel_size
        self.Resnet_version = Resnet_version
        self.Num_residual_blocks = Num_residual_blocks
        self.List_residual_layers = List_residual_layers

    def __call__(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """Build and return the network.
        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.

        Return:
            The output Tensor of the network.
        """
        return self.build_network(inputs, training)

    def build_network(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """Construct the network layers.

        Args:
            inputs: Input tensor.
            training: Training mode flag.

        Returns:
            Constructed network as a tensor.
        """
        inputs = tf.cast(inputs, tf.float32)
        self.training = training
        outputs = self.Start_layer(inputs, self.training)

        for _ in range(len(self.List_residual_layers)):
            self.Num_residual_layers = self.List_residual_layers[_]
            self.Filters = 2 ** (_) * self.First_layer_num_filter
            self.Block_position = _
            for _ in range(self.Num_residual_layers):
                self.Layer_position = _
                outputs = self.Middle_stack(outputs)
        outputs = self.Output_layer(outputs)

        return outputs

    def Start_layer(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """Defines the start layer of the network.

        Args:
            inputs (tf.Tensor): Input tensor to the layer.
            training (bool): Boolean indicating whether the network is in training mode. 

        Returns:
            tf.Tensor: Output tensor after applying the start layer.
        """
        outputs = tf.layers.BatchNormalization(
            traofinable=training,
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
        )(inputs)
        outputs = tf.keras.activations.relu(outputs)
        outputs = tf.keras.layers.Conv2D(
            self.First_layer_num_filter, self.Kernel_size, padding="same"
        )(outputs)

        return outputs

    def Residual_block_V1(
        self, inputs: tf.Tensor, stride: int, training: bool
    ) -> tf.Tensor:
        """Defines a Residual Block Version 1.

        Args:
            inputs (tf.Tensor): Input tensor to the residual block.
            stride (int): Stride size for the convolutional layers.
            training (bool): Boolean indicating whether the network is in training mode.

        Returns:
            tf.Tensor: Output tensor after applying the residual block.
        """
        outputs = tf.layers.BatchNormalization(
            trainable=training,
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
        )(inputs)
        outputs = tf.keras.activations.relu(outputs)
        outputs = tf.keras.layers.Conv2D(
            self.Filters, self.Kernel_size, padding="same", strides=stride
        )(outputs)
        outputs = tf.layers.BatchNormalization(
            trainable=training,
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
        )(outputs)
        outputs = tf.keras.activations.relu(outputs)
        outputs = tf.keras.layers.Conv2D(
            self.Filters, self.Kernel_size, strides=1, padding="same"
        )(outputs)
        if self.Layer_position == 0 and self.Block_position > 0:
            inputs = self.Projection(inputs, self.Filters)

        return tf.add(inputs, outputs)

    def Residual_block_V2(
        self, inputs: tf.Tensor, stride: int, training: bool
    ) -> tf.Tensor:
        """Defines a Residual Block Version 2.

        Args:
            inputs (tf.Tensor): Input tensor to the residual block.
            stride (int): Stride size for the convolutional layers.
            training (bool): Boolean indicating whether the network is in training mode.

        Returns:
            tf.Tensor: Output tensor after applying the residual block.
        """
        outputs = tf.layers.BatchNormalization(
            trainable=training,
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
        )(inputs)
        outputs = tf.keras.activations.relu(outputs)
        outputs = tf.keras.layers.Conv2D(
            self.Filters,
            1,
            strides=stride,
            padding="same",
        )(outputs)
        outputs = tf.layers.BatchNormalization(
            trainable=training,
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
        )(outputs)
        outputs = tf.keras.activations.relu(outputs)
        outputs = tf.keras.layers.Conv2D(
            self.Filters, self.Kernel_size, strides=1, padding="same"
        )(outputs)
        outputs = tf.layers.BatchNormalization(
            trainable=training,
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
        )(outputs)
        outputs = tf.keras.activations.relu(outputs)
        outputs = tf.keras.layers.Conv2D(
            self.Filters * 4, 1, strides=1, padding="same"
        )(outputs)
        inputs = tf.keras.layers.Conv2D(self.Filters * 4, 1, strides=1, padding="same")(
            inputs
        )
        if self.Layer_position == 0 and self.Block_position > 0:
            inputs = self.Projection(inputs, self.Filters * 4)

        return tf.add(inputs, outputs)

    def Middle_stack(self, inputs: tf.Tensor) -> tf.Tensor:
        """Defines the middle stack of layers in the network.

        Args:
            inputs (tf.Tensor): Input tensor to the middle stack.

        Returns:
            tf.Tensor: Output tensor after applying the middle stack.
        """
        if self.Resnet_version == "V1":
            if self.Layer_position == 0 and self.Block_position == 0:
                outputs = self.Residual_block_V1(inputs, 1, self.training)
            elif self.Layer_position == 0 and self.Block_position > 0:
                outputs = self.Residual_block_V1(inputs, 2, self.training)
            else:
                outputs = self.Residual_block_V1(inputs, 1, self.training)
        if self.Resnet_version == "V2":
            if self.Layer_position == 0 and self.Block_position == 0:
                outputs = self.Inception_1(inputs, 1, self.training)
            elif self.Layer_position == 0 and self.Block_position > 0:
                outputs = self.Inception_1(inputs, 2, self.training)
            else:
                outputs = self.Inception_1(inputs, 1, self.training)

        return outputs

    def Projection(self, inputs: tf.Tensor, num_filters: int) -> tf.Tensor:
        """Applies a projection using convolution to match dimensions.

        Args:
            inputs (tf.Tensor): Input tensor.
            num_filters (int): Number of filters for the convolution layer.

        Returns:
            tf.Tensor: Output tensor after projection.
        """
        outputs = tf.keras.layers.Conv2D(
            num_filters, strides=2, padding="same", kernel_size=1
        )(inputs)

        return outputs

    def Projection_dense(self, inputs: tf.Tensor, stride: int) -> tf.Tensor:
        """Defines a dense projection layer.

        Args:
            inputs (tf.Tensor): Input tensor.
            stride (int): Stride size for the convolution.

        Returns:
            tf.Tensor: Output tensor after applying the dense projection.
        """
        outputs = tf.keras.layers.Conv2D(
            int(inputs.shape[-1]),
            strides=stride,
            padding="same",
            kernel_size=self.Kernel_size,
        )(inputs)

        return outputs

    def Output_layer(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Defines the output layer of the network.

        Args:
            inputs (tf.Tensor): Input tensor to the output layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Output tensor and unscaled logits.
        """
        outputs = tf.keras.layers.AveragePooling2D(
            pool_size=inputs.shape[1:3], strides=1
        )(inputs)
        outputs = tf.keras.layers.Dense(1000, use_bias=True)(outputs)
        outputs = tf.keras.layers.Dense(10, use_bias=True)(outputs)
        unscaled_logits = outputs
        outputs = tf.keras.activations.softmax(outputs, axis=-1)
        outputs = tf.squeeze(outputs)

        return outputs, unscaled_logits

    def Inception_1(self, inputs: tf.Tensor, stride: int, training: bool) -> tf.Tensor:
        """Defines an Inception-style block.

        Args:
            inputs (tf.Tensor): Input tensor to the Inception block.
            stride (int): Stride size for convolutional layers.
            training (bool): Boolean indicating whether the network is in training mode.

        Returns:
            tf.Tensor: Output tensor after applying the Inception block.
        """
        if self.Block_position == 0:
            if self.Layer_position == 0:
                self.Filters_1 = self.Filters
                self.inputs_1 = inputs

            outputs = tf.layers.BatchNormalization(
                trainable=training,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
            )(inputs)
            outputs = tf.keras.activations.relu(outputs)

            outputs_1 = tf.keras.layers.Conv2D(
                self.Filters, 1, strides=stride, padding="same"
            )(outputs)

            outputs_2 = tf.keras.layers.Conv2D(
                self.Filters, 1, strides=stride, padding="same"
            )(outputs)
            outputs_2 = tf.layers.BatchNormalization(
                trainable=training,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
            )(outputs_2)
            outputs_2 = tf.keras.activations.relu(outputs_2)
            outputs_2 = tf.keras.layers.Conv2D(
                self.Filters, 3, strides=1, padding="same"
            )(outputs_2)

            outputs_3 = tf.keras.layers.Conv2D(
                self.Filters, 1, strides=stride, padding="same"
            )(outputs)
            outputs_3 = tf.layers.BatchNormalization(
                trainable=training,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
            )(outputs_3)
            outputs_3 = tf.keras.activations.relu(outputs_3)
            outputs_3 = tf.keras.layers.Conv2D(
                self.Filters, 3, strides=1, padding="same"
            )(outputs_3)
            outputs_3 = tf.layers.BatchNormalization(
                trainable=training,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
            )(outputs_3)
            outputs_3 = tf.keras.activations.relu(outputs_3)
            outputs_3 = tf.keras.layers.Conv2D(
                self.Filters, 3, strides=1, padding="same"
            )(outputs_3)

            outputs = [outputs_1, outputs_2, outputs_3]
            outputs = tf.keras.layers.Concatenate(axis=-1)(outputs)
            outputs = tf.keras.layers.Conv2D(
                self.Filters * 4, 1, strides=1, padding="same"
            )(outputs)
            inputs = tf.keras.layers.Conv2D(
                self.Filters * 4, 1, strides=1, padding="same"
            )(inputs)
            if self.Layer_position == 0 and self.Block_position > 0:
                inputs = self.Projection(inputs, self.Filters * 4)

        if self.Block_position == 1:
            if self.Layer_position == 0:
                self.Filters = self.Filters + self.Filters_1
                self.Filters_2 = self.Filters + self.Filters_1
                inputs = tf.keras.layers.Concatenate(axis=-1)([self.inputs_1, inputs])
                self.inputs_2 = inputs

            outputs = tf.layers.BatchNormalization(
                trainable=training,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
            )(inputs)
            outputs = tf.keras.activations.relu(outputs)

            outputs_1 = tf.keras.layers.Conv2D(
                self.Filters, 1, strides=stride, padding="same"
            )(outputs)

            outputs_3 = tf.keras.layers.Conv2D(
                self.Filters, kernel_size=1, strides=stride, padding="same"
            )(outputs)
            outputs_3 = tf.layers.BatchNormalization(
                trainable=training,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
            )(outputs_3)
            outputs_3 = tf.keras.activations.relu(outputs_3)
            outputs_3 = tf.keras.layers.Conv2D(
                self.Filters, kernel_size=(1, 5), strides=1, padding="same"
            )(outputs_3)
            outputs_3 = tf.layers.BatchNormalization(
                trainable=training,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
            )(outputs_3)
            outputs_3 = tf.keras.activations.relu(outputs_3)
            outputs_3 = tf.keras.layers.Conv2D(
                self.Filters, kernel_size=(5, 1), strides=1, padding="same"
            )(outputs_3)

            outputs = [outputs_1, outputs_3]
            outputs = tf.keras.layers.Concatenate(axis=-1)(outputs)
            outputs = tf.keras.layers.Conv2D(
                self.Filters * 4, 1, strides=1, padding="same"
            )(outputs)
            inputs = tf.keras.layers.Conv2D(
                self.Filters * 4, 1, strides=1, padding="same"
            )(inputs)
            if self.Layer_position == 0 and self.Block_position > 0:
                inputs = self.Projection(inputs, self.Filters * 4)

        if self.Block_position > 1:
            if self.Layer_position == 0:
                self.Filters = self.Filters + self.Filters_1 + self.Filters_2
                self.inputs_2 = self.Projection_dense(self.inputs_2, 2)
                self.inputs_1 = self.Projection_dense(self.inputs_1, 2)
                inputs = tf.keras.layers.Concatenate(axis=-1)(
                    [self.inputs_2, self.inputs_1, inputs]
                )

            outputs = tf.layers.BatchNormalization(
                trainable=training,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
            )(inputs)
            outputs = tf.keras.activations.relu(outputs)

            outputs_1 = tf.keras.layers.Conv2D(
                self.Filters, 1, strides=stride, padding="same"
            )(outputs)

            outputs_3 = tf.keras.layers.Conv2D(
                self.Filters, kernel_size=1, strides=stride, padding="same"
            )(outputs)
            outputs_3 = tf.layers.BatchNormalization(
                trainable=training,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
            )(outputs_3)
            outputs_3 = tf.keras.activations.relu(outputs_3)
            outputs_3 = tf.keras.layers.Conv2D(
                self.Filters, kernel_size=(1, 3), strides=1, padding="same"
            )(outputs_3)
            outputs_3 = tf.layers.BatchNormalization(
                trainable=training,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
            )(outputs_3)
            outputs_3 = tf.keras.activations.relu(outputs_3)
            outputs_3 = tf.keras.layers.Conv2D(
                self.Filters, kernel_size=(3, 1), strides=1, padding="same"
            )(outputs_3)

            outputs = [outputs_1, outputs_3]
            outputs = tf.keras.layers.Concatenate(axis=-1)(outputs)
            outputs = tf.keras.layers.Conv2D(
                self.Filters * 4, 1, strides=1, padding="same"
            )(outputs)
            inputs = tf.keras.layers.Conv2D(
                self.Filters * 4, 1, strides=1, padding="same"
            )(inputs)
            if self.Layer_position == 0 and self.Block_position > 0:
                inputs = self.Projection(inputs, self.Filters * 4)

        return tf.add(inputs, outputs)
