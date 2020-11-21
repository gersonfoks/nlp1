from practical_2.utils import load_history, plot_history

name = "attention_pos"
history = load_history('histories/{}'.format(name))

plot_history(history, name)