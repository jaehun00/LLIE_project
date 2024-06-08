import matplotlib.pyplot as plt
import os
import pickle
import argparse

parser = argparse.ArgumentParser("PLOT")
parser.add_argument('--train', type=str, default='Train', help='batch size')
args = parser.parse_args()

path = args.train


def plot_loss(epochs, fidelity_loss, smooth_loss, exp_loss, col_loss):
    plt.figure(figsize=(10, 8))

    # fidelity_loss 그래프
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs+1), fidelity_loss, label='fidelity_loss', marker='', color='r', linewidth=2)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('SCI model - Fidelity Loss')
    plt.legend()
    plt.grid()

    # smooth_loss 그래프
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs+1), smooth_loss, label='smooth_loss', marker='', color='b', linewidth=2)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('SCI model - Smooth Loss')
    plt.legend()
    plt.grid()

    # exp_loss 그래프
    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs+1), exp_loss, label='exp_loss', marker='', color='y', linewidth=2)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('SCI model - Exposure Loss')
    plt.legend()
    plt.grid()

    # col_loss 그래프
    plt.subplot(2, 2, 4)
    plt.plot(range(1, epochs+1), col_loss, label='col_loss', marker='', color='g', linewidth=2)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('SCI model - Color Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()  # 각각의 subplot 간격 조절
    plt.show()

# plot()
def plot_trainval(epochs, train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_loss, label='train_loss', marker='', color='r', linewidth = 2)
    plt.plot(range(1, epochs+1), val_loss, label='val_loss', marker='', color='b', linewidth = 2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim([1, epochs])
    plt.ylim([0, 10])
    plt.title('SCI model')
    plt.legend()
    plt.grid()
    #os.makedirs('./plot/', exist_ok=True)
    #plt.savefig('./plot/weight_188.png')
    plt.show()
# pickle 파일 경로
train_loss_path = path + '/train_losses.pkl'
val_loss_path = path + '/val_losses.pkl'

# pickle 파일 읽기
with open(train_loss_path, 'rb') as file1:
    train_data = pickle.load(file1)

with open(val_loss_path, 'rb') as file2:
    val_data = pickle.load(file2)

epochs = len(train_data)
train_loss = train_data
val_loss = val_data

plot_trainval(epochs, train_loss, val_loss)

# pickle 파일 경로
fidelity_loss_path = path + '/train_loss/fidelity_losses.pkl'
smooth_loss_path = path + '/train_loss/smooth_losses.pkl'
exp_loss_path = path + '/train_loss/exp_losses.pkl'
col_loss_path = path + '/train_loss/col_losses.pkl'
# pickle 파일 읽기
with open(fidelity_loss_path, 'rb') as file1:
    fidelity_data = pickle.load(file1)

with open(smooth_loss_path, 'rb') as file2:
    smooth_data = pickle.load(file2)

with open(exp_loss_path, 'rb') as file1:
    exp_data = pickle.load(file1)

with open(col_loss_path, 'rb') as file2:
    col_data = pickle.load(file2)

epochs = len(fidelity_data)

plot_loss(epochs, fidelity_data, smooth_data, exp_data, col_data)

# pickle 파일 경로
fidelity_loss_path = path + '/val_loss/fidelity_losses.pkl'
smooth_loss_path = path + '/val_loss/smooth_losses.pkl'
exp_loss_path = path + '/val_loss/exp_losses.pkl'
col_loss_path = path + '/val_loss/col_losses.pkl'
# pickle 파일 읽기
with open(fidelity_loss_path, 'rb') as file1:
    fidelity_data = pickle.load(file1)

with open(smooth_loss_path, 'rb') as file2:
    smooth_data = pickle.load(file2)

with open(exp_loss_path, 'rb') as file1:
    exp_data = pickle.load(file1)

with open(col_loss_path, 'rb') as file2:
    col_data = pickle.load(file2)


epochs = len(fidelity_data)

plot_loss(epochs, fidelity_data, smooth_data, exp_data, col_data)
