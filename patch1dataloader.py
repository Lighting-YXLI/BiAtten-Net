# coding=gbk
from PIL import Image
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
class MyDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, images_path1: list,images_path2: list, images_class: list, transform1=None,transform2=None):
        self.images_path1 = images_path1
        self.images_path2 = images_path2
        self.images_class = images_class
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.images_path1)

    def __getitem__(self, item):
        img1 = Image.open(self.images_path1[item])
        img2 = Image.open(self.images_path2[item])
        # RGB为彩色图片，L为灰度图片
        if img1.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path1[item]))
        label = self.images_class[item]

        if self.transform1 is not None:
            img1 = self.transform1(img1)
            img2 = self.transform2(img2)

        return img1, img2, label

#    @staticmethod
#    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
#        images, labels = tuple(zip(*batch))
#        images = torch.stack(images, dim=0)
#        labels = torch.as_tensor(labels)
#        return images, labels

def creat_path(txt):
    with open(txt, 'r') as fh:
        imgs_path = []
        imgs_label=[]
        for line in fh:
            line = line.strip('\n')  # 移除字符串首尾的换行符
            line = line.rstrip()  # 删除末尾空
            words = line.split()  # 以空格为分隔符 将字符串分成
            imgs_path.append(words[1])
            imgs_label.append(float(words[0]))
    return imgs_path,imgs_label

#train_lbp_path,train_imgs_label=creat_path('G:/liyx/DeepSRQ-master/QADS/lbp_patches/train_lbp_patches.txt')
#train_structure_path,_=creat_path('G:/liyx/DeepSRQ-master/QADS/structure_patches/train_structure_patches.txt')
#val_lbp_path,val_imgs_label=creat_path('G:/liyx/DeepSRQ-master/QADS/lbp_patches/test_lbp_patches.txt')
#val_structure_path,_=creat_path('G:/liyx/DeepSRQ-master/QADS/structure_patches/test_structure_patches.txt')

train_lbp_path,train_imgs_label=creat_path('C:/D/BASENet/DeepSRQ-master/CVIU_reference_patch/orgHRtrain.txt')
train_structure_path,_=creat_path('C:/D/BASENet/DeepSRQ-master/CVIU_train_patch/orgSRtrain.txt')
val_lbp_path,val_imgs_label=creat_path('C:/D/BASENet/DeepSRQ-master/SISAR/test_HR_patch.txt')
val_structure_path,_=creat_path('C:/D/BASENet/DeepSRQ-master/SISAR/test_SISAR_patch.txt')


#print(len(train_imgs_path))
#print('---------------------')
#print(len(train_imgs_label))
data_transform = {
        "train_lbp": transforms.Compose([#transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()
                                     #transforms.Normalize([0.45255446, 0.45361045, 0.37407586], [0.11673171, 0.109770544, 0.10265977])
               ]),

        "train_structure": transforms.Compose([  # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        #transforms.Normalize([0.45175493, 0.45306018, 0.37391272], [0.09589117, 0.08915779, 0.08248971])
            ]),

        "val_structure": transforms.Compose([#transforms.Resize(256),
                                   #transforms.CenterCrop(224),
                                   transforms.ToTensor()
                                   #transforms.Normalize([0.4604404, 0.4595839, 0.3750678], [0.09742115, 0.09028603, 0.08221073])
            ]),

        "val_lbp": transforms.Compose([#transforms.Resize(256),
                                   #transforms.CenterCrop(224),
                                   transforms.ToTensor()
                                   #transforms.Normalize([0.4612089, 0.4601845, 0.37503883], [0.11930786, 0.111974746, 0.103294864])
            ])}


#实例化训练数据集
#batch_size = 128
nw = 0
train_dataset = MyDataSet(images_path1=train_lbp_path,
                          images_path2=train_structure_path,
                          images_class=train_imgs_label,
                          transform1=data_transform["train_lbp"],
                          transform2=data_transform["train_structure"]
                          )
#print(len(train_dataset))
# 实例化验证数据集
val_dataset = MyDataSet(images_path1=val_lbp_path,
                        images_path2=val_structure_path,
                        images_class=val_imgs_label,
                        transform1=data_transform["val_lbp"],
                        transform2=data_transform["val_structure"],
                        )
train_loader = data.DataLoader(train_dataset,
                               batch_size=128,
                               shuffle=True,
                               pin_memory=True,
                               num_workers=nw,
                               )

val_loader = data.DataLoader(val_dataset,
                             batch_size=128,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=nw,
                             )

#for epoch in range(2):
    # 必须加括号！！否则报错ValueError: not enough values to unpack (expected 3, got 2)
#    for i, (inputs1,inputs2, labels) in enumerate(train_loader):

#        print("epoch：", epoch, "的第", i, "个inputs1 length{}, inputs2 length{},list中第0个shape{}".format(
#            len(inputs1), len(inputs2),inputs1[0].shape))

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean1 = torch.zeros(6)
    std1 = torch.zeros(6)
    mean2 = torch.zeros(6)
    std2 = torch.zeros(6)
    i=0
    for X1,X2, _ in train_loader:
        print(i)
        i+=1
        for d in range(3):
            mean1[d] += X1[:, d, :, :].mean()
            std1[d] += X1[:, d, :, :].std()
            mean2[d] += X2[:, d, :, :].mean()
            std2[d] += X2[:, d, :, :].std()
    mean1.div_(len(train_data))
    std1.div_(len(train_data))
    mean2.div_(len(train_data))
    std2.div_(len(train_data))
    return list(mean1.numpy()), list(std1.numpy()), list(mean2.numpy()), list(std2.numpy())

print(getStat(val_dataset))