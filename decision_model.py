import torch 
import torch.nn as nn
import os
import json
from torchvision import transforms
from PIL import Image
from tools import get_md5


initial_task_name_list = ['000', '002', '003', '004', '006', 
                            '010', '012', '013', '014', '016',
                            '020', '021', '022', '023', '024', '025', '026',
                            '034',
                            '100', 
                            '110', '112', '114', 
                            '120', '122',
                            '200', '202', '204',
                            '210', '211', '212', '213', '214', '216',
                            '220', '222', '224', 
                            '300', '302', '304', 
                            '310', '312', '313', '314',
                            '320', '322', '324',  
                            '400', '402', '404', 
                            '410', '411', '412', '413', '414', '415',
                            '420', '422', '424', '426', 
                            '436',
                            '500', '502', '503', '504', 
                            '510', '512', '513', '514', 
                            '520', '522', '523', '524', '526', 
                            '600', '602', '604', 
                            '610', '612', '613', '614', '616',
                            '620', '622', '624', '625', '626', 
                            '630', '632', '633', '634', '635', '636'
                            ]


# initial_task_name_list = [
#         '003-013-023-213-313-413-503-513-523-613-633',
#         '120-122-420-422-424-426-520-522-523-524-526',
#         '020-021-022-023-024-025-026-220-222-224-320-322-324',
#         '010-012-013-014-016',
#         '002-012-022-112-122-202-212-222-302-312-322-402-412-422-502-512-522-602-612-622-632',
#         '100-110-112-114-400-402-404-410-411-412-413-414-415-500-502-503-504-510-512-513-514',
#         '004-014-024-034-114-204-214-224-304-314-324-404-414-424-504-514-524-604-614-624-634',
#         '000-010-020-100-110-120-200-210-210-220-220-300-310-320-400-410-420-500-510-520-600-610-620-630',
#         '000-002-003-004-006-200-202-204-210-211-212-213-214-216-300-302-304-310-312-313-314',
#     ]


# Device configuration
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 20
num_classes = len(initial_task_name_list)
batch_size = 512
learning_rate = 0.001


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(100352, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        # print(path)
        img = Image.open(f)
        return img.convert('RGB')


# TODO add dataset
class Scenario(torch.utils.data.Dataset):
    def __init__(self, root, tree_index_list):

        self.data_list = []
        self.label_list = []

        self.transform = transforms.Compose(
                        [transforms.Resize([224, 224]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
        
        for data_type in ['train', 'val']:
            for class_index, tree_index in enumerate(tree_index_list):
                json_path = os.path.join('data/bdd_soda_traffic/annotations/sub_bdd', data_type, get_md5(tree_index)+'.json')
                
                with open(json_path,'r') as load_f:
                    load_dict = json.load(load_f)

                for image in load_dict['images']:
                    image_path = os.path.join(root, data_type, 'images', image['file_name'])
                    self.data_list.append(image_path)
                    self.label_list.append(class_index)

        # data_index = list(data_index)
        # for index in range(len(data_index)):
        #     data_index[index] = int(data_index[index])

        # for index, item in enumerate(data.keys()):
        #     if index in data_index:
        #         for data_each in data[item]['data']:
        #             data_list.append(data_each)
        #         for label_each in data[item]['label']:
        #             label_list.append(label_each)

        # self.data_list = torch.from_numpy(np.array(data_list))
        # self.label_list = torch.from_numpy(np.array(label_list))

        self.classes = [i for i in range(len(tree_index_list))]

    def __getitem__(self, item):
        image = self.transform(pil_loader(self.data_list[item]))
        return image, self.label_list[item]

    def __len__(self):
        return len(self.data_list)


def train_test(tree_index_list):

    classification_data = Scenario('data/bdd_soda_traffic', tree_index_list)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=classification_data,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=classification_data,
                                            batch_size=batch_size, 
                                            shuffle=False)

    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')


if __name__ == '__main__':
    # tree_index_list = [
    #     '003-013-023-213-313-413-503-513-523-613-633',
    #     '120-122-420-422-424-426-520-522-523-524-526',
    #     '020-021-022-023-024-025-026-220-222-224-320-322-324',
    #     '010-012-013-014-016',
    #     '002-012-022-112-122-202-212-222-302-312-322-402-412-422-502-512-522-602-612-622-632',
    #     '100-110-112-114-400-402-404-410-411-412-413-414-415-500-502-503-504-510-512-513-514',
    #     '004-014-024-034-114-204-214-224-304-314-324-404-414-424-504-514-524-604-614-624-634',
    #     '000-010-020-100-110-120-200-210-210-220-220-300-310-320-400-410-420-500-510-520-600-610-620-630',
    #     '000-002-003-004-006-200-202-204-210-211-212-213-214-216-300-302-304-310-312-313-314',
    # ]

    train_test(initial_task_name_list)
