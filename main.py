
if __name__ == "__main__":
    op = open('dataset/mpii/train_joints.csv', 'r')
    dataes = op.readlines()
    print(len(dataes))
    for data in dataes:
        print(data.split(',')[0].split('/')[-1].split('.')[0])
