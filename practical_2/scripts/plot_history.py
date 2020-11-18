from practical_2.utils import load_history, plot_history

name = "w2v_pt_deep_cbow"
history = load_history('histories/{}'.format(name))

plot_history(history, name)