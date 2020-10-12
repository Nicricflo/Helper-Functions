import plotly.express as px
from keras.callbacks import Callback
from IPython.display import clear_output


class plotly_callback(Callback):

    def __init__(self):

        self.i = 1
        self.x = []
        self.losses = []
        self.val_losses = []
        self.val_accs = []
        self.acc_avg = []
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_accuracy'))
        self.acc_avg.append(logs.get('accuracy'))

        self.i += 1

        clear_output(wait=True)

        fig = px.scatter(x=self.x, y=self.acc_avg, labels={
                         'x': 'epoch', 'y': 'train accuracy'}, template='ggplot2', render_mode='auto')
        fig.data[0].update(mode='markers+lines')

        fig2 = px.scatter(x=self.x, y=self.val_accs, labels={
                          'x': 'epoch', 'y': 'val accuracy'})
        fig2.data[0].update(mode='markers+lines')

        fig.add_trace(fig2.data[0])

        fig.show()


if __name__ == '__main__':
    pass
