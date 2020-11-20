from practical_2.utils import load_history, plot_history

name = "lstm"
history = load_history('histories/{}'.format(name))

plot_history(history, name)