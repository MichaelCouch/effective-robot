#!/usr/bin/env python
import pika, sys, os
import pyglet
from pyglet import shapes, clock
import json
import numpy as np

class NeuralNetVisualizer:
    """Visualizes the neural network, receiving updates of state based on message queue"""

    def __init__(self):
        self._connection = pika.BlockingConnection()
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue='neuralnet')

        """Initialize a pyglet window"""
        self._width = 640
        self._height = 480
        self._window = pyglet.window.Window(self._width, self._height)
        self._batch = pyglet.graphics.Batch()

        clock.schedule(self._update, 0.01)
        self.n = 0

    def run(self):
        pyglet.app.run()

    def _get_message_body(self):
        method_frame, header_frame, body = self._channel.basic_get('neuralnet')
        if method_frame:
            self._channel.basic_ack(method_frame.delivery_tag)
            return json.loads(body.decode())
        else:
            return None

    def _update_image(self, message):
        """Update the image
        :message: dictionary of layers, and weights
        :returns: None

        """
        self.n += 1
        layers = message['layers']
        weights = message['weights']
        nlayers = len(layers)
        neurons = []
        for i, layer in enumerate(layers):
            x = int(self._width * (i + 1) / (nlayers + 1))
            if i == nlayers - 1:
                m = sum([np.exp(a[0]) for a in layer])
            else:
                m = np.exp(max([abs(a[0]) for a in layer]))
            for j, neuron in enumerate(layer):
                y = int(self._height * (j + 1) / (len(layer) + 1))
                if i == nlayers - 1:
                    neurons.append(shapes.Circle(x, y, np.exp(neuron[0]) / m * 100, color=(127 + int(neuron[0] / m * 128) ,127 - int(neuron[0] / m * 128), 0), batch=self._batch))
                else:
                    neurons.append(shapes.Circle(x, y, np.exp(abs(neuron[0])) / m * 10, color=(127 + int(neuron[0] / m * 128) ,127 - int(neuron[0] / m * 128), 0), batch=self._batch))
        lines = []
        for i, weight in enumerate(weights):
            xl = int(self._width * (i + 1) / (nlayers + 1))
            xr = int(self._width * (i + 2) / (nlayers + 1))
            for ji, links in enumerate(weight):
                m = max([max([abs(a) for a in link]) for link in weight])
                yr = int(self._height * (ji + 1) / (len(weight) + 1))
                for jo, link in enumerate(links):
                    yl = int(self._height * (jo + 1) / (len(links) + 1))
                    lines.append(shapes.Line(xl, yl, xr, yr, width=1, color=(127 + int(link / m * 128) ,127 - int(link / m * 128), 0), batch=self._batch))



        self._window.clear()
        self._batch.draw()

    def _update(self, dt, *args, **kwargs):
        message = self._get_message_body()
        if message:
            self._update_image(message)


def main():
    vis = NeuralNetVisualizer()
    vis.run()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


