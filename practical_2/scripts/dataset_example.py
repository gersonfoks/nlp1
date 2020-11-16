from practical_2.TreeDataset import TreeDataset

train_dataset = TreeDataset("trees/train.txt")


v = train_dataset.v
print(v.w2i['century'])