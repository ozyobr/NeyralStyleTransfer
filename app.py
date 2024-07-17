import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import copy

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

cnn = models.vgg19(pretrained=True).features.to(torch.device("cpu")).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(torch.device("cpu"))
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(torch.device("cpu"))

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def image_loader(image):
    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = loader(image).unsqueeze(0)
    return image.to(torch.device("cpu"), torch.float)

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(torch.device("cpu"))

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    progress_bar = st.progress(0)

    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            progress_bar.progress(min(run[0] / num_steps, 1.0))

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img

st.title("Neural Style Transfer")

col1, col2 = st.columns(2)

with col1:
    content_image = st.file_uploader("Загрузите контент-изображение", type=["jpg", "jpeg", "png"])

with col2:
    style_image = st.file_uploader("Загрузите стиль-изображение", type=["jpg", "jpeg", "png"])

if content_image and style_image:
    content_image = Image.open(content_image)
    style_image = Image.open(style_image)

    col1.image(content_image, caption='Контент-изображение', use_column_width=True)
    col2.image(style_image, caption='Стиль-изображение', use_column_width=True)

    content_tensor = image_loader(content_image)
    style_tensor = image_loader(style_image)
    input_tensor = content_tensor.clone()

    if st.button("Запустить перенос стиля"):
        with st.spinner('Выполняется стилизация...'):
            output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                        content_tensor, style_tensor, input_tensor, num_steps=300)

        output_image = output.cpu().clone()
        output_image = output_image.squeeze(0)
        output_image = transforms.ToPILImage()(output_image)

        st.markdown("<h2 style='text-align: center;'>Результат стилизации</h2>", unsafe_allow_html=True)
        st.image(output_image, caption='Результат стилизации', use_column_width=True, clamp=True)
