import torch.nn as nn
import torch 
from submodules.adain.utils.Style_function import adaptive_instance_normalization as adain
from submodules.adain.utils.Style_function import calc_mean_std, styleLoss, GramMatrix, calc_cov

from submodules.adain.utils.whitening_core import whitening, feature_wct

# decoder = nn.Sequential(
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 256, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 3, (3, 3)),
# )

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

# vgg = nn.Sequential(
#     nn.Conv2d(3, 3, (1, 1)),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(3, 64, (3, 3)),
#     nn.ReLU(),  # relu1-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),  # relu1-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 128, (3, 3)),
#     nn.ReLU(),  # relu2-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),  # relu2-2
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 256, (3, 3)),
#     nn.ReLU(),  # relu3-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),  # relu3-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 512, (3, 3)),
#     nn.ReLU(),  # relu4-1, this is the last layer used
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu4-4
#     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-1
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-2
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU(),  # relu5-3
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 512, (3, 3)),
#     nn.ReLU()  # relu5-4
# )



class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.inr_enc_64 = nn.Sequential(*enc_layers[:4])
        self.inr_enc_128 = nn.Sequential(*enc_layers[:11])
        self.inr_enc = nn.Sequential(*enc_layers[:18])
        self.inr_enc_256 = nn.Sequential(*enc_layers[:18])
        # self.inr_enc_deep = nn.Sequential(*enc_layers[:26])
        self.inr_enc_deep = nn.Sequential(*enc_layers[:31])
        self.inr_enc_512 = nn.Sequential(*enc_layers[:31])


        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_for_inr_decoder(self, input):
        out_1 = self.inr_enc(input)
        out = self.upsample_4(out_1)
        return out

    def encode_for_inr_decoder_64(self, input):
        out_1 = self.inr_enc_64(input)
        # out = self.upsample(out_1)
        return out_1
    
    def encode_for_inr_decoder_128(self, input):
        out_1 = self.inr_enc_128(input)
        out = self.upsample(out_1)
        return out
    
    def encode_for_inr_decoder_256(self, input):
        out_1 = self.inr_enc_256(input)
        out_2 = self.upsample(out_1)
        out = self.upsample(out_2)
        return out
    
    def encode_for_inr_decoder_512(self, input):
        out_1 = self.inr_enc_512(input)
        out_2 = self.upsample(out_1)
        out_3 = self.upsample(out_2)
        out = self.upsample(out_3)
        return out
    
    def encode_for_inr_decoder_128_half(self, input):
        out_1 = self.inr_enc_128(input)
        out = self.upsample(out_1)
        out_1d = out.reshape(out.shape[0], 128, -1)
        mean = torch.mean(out_1d, -1, keepdim=True)
        out_1d = (out_1d - mean)/2 + mean
        out = out_1d.reshape(out.shape)
        return out

    def encode_for_inr_decoder_128_down(self, input):
        out_1 = self.inr_enc_128(input)
        # out = self.upsample(out_1)
        return out_1

    def encode_for_inr_decoder_deep(self, input):
        out_1 = self.inr_enc_deep(input)
        # out = self.upsample_4(out_1)
        out = self.upsample_8(out_1)
        return out

    def encode_for_inr_decoder_same_size(self, input):
        out = self.inr_enc(input)
        # out = self.upsample(out_1)
        return out

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def encode_with_intermediate_half(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            feat = func(results[-1])
            feat_1d = feat.reshape(feat.shape[0], feat.shape[1], -1)
            mean = torch.mean(feat_1d, -1, keepdim=True)
            feat_1d = (feat_1d - mean)/2 + mean
            feat = feat_1d.reshape(feat.shape)

            results.append(feat)
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input
    
    def encode_mean_std(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        input_mean, input_std = calc_mean_std(input)
        return input_mean, input_std

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        # assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def calc_style_loss_wct(self, input, target):
        # assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_cov = calc_cov(input)
        target_cov = calc_cov(target)
        return self.mse_loss(input_cov, target_cov)
            #    self.mse_loss(input_std, target_std)

    def calc_style_loss_gram(self, inputs, targets):
        style_loss = 0
        for i in range(len(inputs)):
            input, target = inputs[i], targets[i]
            style_loss += styleLoss(input, target)
        return style_loss

    def calc_nerf_loss(self, x, content_gt, style_gt):
        fea_x = self.encode_with_intermediate(x)
        fea_style_gt = self.encode_with_intermediate(style_gt)
        fea_content_gt = self.encode_with_intermediate(content_gt)
        # loss_s = self.calc_style_loss(fea_x[0], fea_style_gt[0])
        loss_s = self.calc_style_loss_gram(fea_x, fea_style_gt)
        loss_c = self.calc_content_loss(fea_x[-1], fea_content_gt[-1])
        return loss_c, loss_s

    def forward(self, content, style, alpha=1.0, return_stylized_content=False):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)


        stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        stylized_content = self.decoder(stylized_content_feat)
        stylized_content_feat_encoded = self.encode_with_intermediate(stylized_content)

        loss_c = self.calc_content_loss(stylized_content_feat_encoded[-1], stylized_content_feat)
        loss_s = self.calc_style_loss(stylized_content_feat_encoded[0], style_feats[0])
        
        
        
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content
        else:
            return loss_c, loss_s

    def get_inr_style_loss(self, transferred_im, style):
        # assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)

        stylized_content_feat_encoded = self.encode_with_intermediate(transferred_im)

        loss_s = self.calc_style_loss(stylized_content_feat_encoded[0], style_feats[0])
        
        
        
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded[i], style_feats[i])
        return loss_s


    def inr_adain_deep(self, content, style, inr_decoder, alpha=1.0, return_stylized_content=False):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        adain_style_feat = self.encode_for_inr_decoder_deep(style)
        content_feat = self.encode_for_inr_decoder_deep(content)


        # stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = adain(content_feat, adain_style_feat)
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        stylized_content = inr_decoder(stylized_content_feat).type(torch.float32)
        stylized_content_feat_encoded = self.encode_for_inr_decoder_deep(stylized_content)

        stylized_content_feat_encoded_style_loss = self.encode_with_intermediate(stylized_content)

        loss_c = self.calc_content_loss(stylized_content_feat_encoded, stylized_content_feat)
        loss_s = self.calc_style_loss(stylized_content_feat_encoded_style_loss[0], style_feats[0])
        
        
        
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded_style_loss[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content
        else:
            return loss_c, loss_s
        
    def inr_adain_128_guided(self, content, style, inr_decoder, alpha=1.0, return_stylized_content=False, GF = None):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        adain_style_feat = self.encode_for_inr_decoder_128(style)
        tch_img = style
        tch_mask = adain_style_feat

        out = GF(tch_mask, tch_img)
        adain_style_feat = out

        content_feat = self.encode_for_inr_decoder_128(content)
        tch_img = content
        tch_mask = content_feat

        out = GF(tch_mask, tch_img)
        content_feat = out


        # stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = adain(content_feat, adain_style_feat)
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        # tch_img = content
        # tch_mask = stylized_content_feat

        # out = GF(tch_mask, tch_img)
        # stylized_content_feat = out

        # stylized_content_feat = whitening(stylized_content_feat)

        stylized_content = inr_decoder(stylized_content_feat).type(torch.float32)
        stylized_content_feat_encoded = self.encode_for_inr_decoder_128(stylized_content)
        tch_img = stylized_content
        tch_mask = stylized_content_feat_encoded

        out = GF(tch_mask, tch_img)
        stylized_content_feat_encoded = out


        stylized_content_feat_encoded_style_loss = self.encode_with_intermediate(stylized_content)

        loss_c = self.calc_content_loss(stylized_content_feat_encoded, stylized_content_feat)
        loss_s = self.calc_style_loss(stylized_content_feat_encoded_style_loss[0], style_feats[0])
        
        
        
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded_style_loss[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content
        else:
            return loss_c, loss_s
        
    def inr_adain_128_guided3(self, content, style, inr_decoder, alpha=1.0, return_stylized_content=False, GF = None):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        adain_style_feat = self.encode_for_inr_decoder_128(style)
        tch_img = style
        tch_mask = adain_style_feat

        out = GF(tch_mask, tch_img)
        adain_style_feat_guided = out

        content_feat = self.encode_for_inr_decoder_128(content)
        tch_img = content
        tch_mask = content_feat

        out = GF(tch_mask, tch_img)
        content_feat_guided = out


        # stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = adain(content_feat, adain_style_feat)
        stylized_content_feat_guided = adain(content_feat_guided, adain_style_feat_guided)
        # stylized_content_feat = alpha * stylized_content_feat_guided + (1 - alpha) * content_feat

        # tch_img = content
        # tch_mask = stylized_content_feat

        # out = GF(tch_mask, tch_img)
        # stylized_content_feat = out

        # stylized_content_feat = whitening(stylized_content_feat)

        stylized_content = inr_decoder(stylized_content_feat_guided).type(torch.float32)
        stylized_content_feat_encoded = self.encode_for_inr_decoder_128(stylized_content)
        # tch_img = stylized_content
        # tch_mask = stylized_content_feat_encoded

        # out = GF(tch_mask, tch_img)
        # stylized_content_feat_encoded = out


        stylized_content_feat_encoded_style_loss = self.encode_with_intermediate(stylized_content)

        loss_c = self.calc_content_loss(stylized_content_feat_encoded, stylized_content_feat)
        loss_s = self.calc_style_loss(stylized_content_feat_encoded_style_loss[0], style_feats[0])
        
        
        
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded_style_loss[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content
        else:
            return loss_c, loss_s

    def inr_adain_128(self, content, style, inr_decoder, alpha=1.0, return_stylized_content=False, GF = None):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        adain_style_feat = self.encode_for_inr_decoder_128(style)
        content_feat = self.encode_for_inr_decoder_128(content)


        # stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = adain(content_feat, adain_style_feat)
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        # stylized_content_feat = whitening(stylized_content_feat)

        stylized_content = inr_decoder(stylized_content_feat).type(torch.float32)
        stylized_content_feat_encoded = self.encode_for_inr_decoder_128(stylized_content)

        stylized_content_feat_encoded_style_loss = self.encode_with_intermediate(stylized_content)

        loss_c = self.calc_content_loss(stylized_content_feat_encoded, stylized_content_feat)
        loss_s = self.calc_style_loss(stylized_content_feat_encoded_style_loss[0], style_feats[0])
        
        
        
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded_style_loss[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content
        else:
            return loss_c, loss_s

    def inr_adain_128_half(self, content, style, inr_decoder, alpha=1.0, return_stylized_content=False, GF = None):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        adain_style_feat = self.encode_for_inr_decoder_128_half(style)
        content_feat = self.encode_for_inr_decoder_128_half(content)
        content_feat_loss = self.encode_for_inr_decoder_128(content)



        # stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = adain(content_feat, adain_style_feat)
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        # stylized_content_feat = whitening(stylized_content_feat)

        stylized_content = inr_decoder(stylized_content_feat).type(torch.float32)
        stylized_content_feat_encoded = self.encode_for_inr_decoder_128_half(stylized_content)

        stylized_content_feat_encoded_style_loss = self.encode_with_intermediate(stylized_content)

        loss_c = self.calc_content_loss(stylized_content_feat_encoded, stylized_content_feat)
        loss_s = self.calc_style_loss(stylized_content_feat_encoded_style_loss[0], style_feats[0])
        
        
        
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded_style_loss[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content
        else:
            return loss_c, loss_s

        
    def inr_adain_128_relu2(self, content, style, inr_decoder, alpha=1.0, return_stylized_content=False):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        adain_style_feat = self.encode_for_inr_decoder_128(style)
        content_feat = self.encode_for_inr_decoder_128(content)


        # stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = adain(content_feat, adain_style_feat)
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        # stylized_content_feat = whitening(stylized_content_feat)

        stylized_content = inr_decoder(stylized_content_feat).type(torch.float32)
        stylized_content_feat_encoded = self.encode_for_inr_decoder_128(stylized_content)

        stylized_content_feat_encoded_style_loss = self.encode_with_intermediate(stylized_content)

        loss_c = self.calc_content_loss(stylized_content_feat_encoded, stylized_content_feat)
        loss_s = self.calc_style_loss(stylized_content_feat_encoded_style_loss[0], style_feats[0])
        
        for i in range(1, 2):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded_style_loss[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content
        else:
            return loss_c, loss_s
        
    def inr_wct(self, content, style, inr_decoder, alpha=1.0, return_stylized_content=False):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        adain_style_feat = self.encode_for_inr_decoder_64(style)
        content_feat = self.encode_for_inr_decoder_64(content)

        loss_c = 0
        loss_s = 0

        # stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = feature_wct(content_feat, adain_style_feat)
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        # stylized_content_feat = whitening(stylized_content_feat)

        stylized_content = inr_decoder(stylized_content_feat).type(torch.float32)
        stylized_content_feat_encoded = self.encode_for_inr_decoder_64(stylized_content)

        stylized_content_feat_encoded_style_loss = self.encode_with_intermediate(stylized_content)

        # loss_c = self.calc_content_loss(stylized_content_feat_encoded, stylized_content_feat)
        # loss_s = self.calc_style_loss_wct(stylized_content_feat_encoded_style_loss[0], style_feats[0])
        
        
        
        # for i in range(1, 4):
        #     loss_s += self.calc_style_loss_wct(stylized_content_feat_encoded_style_loss[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content
        else:
            return loss_c, loss_s
        
        
    def inr_adain_128_recon_loss(self, content, style, inr_decoder, alpha=1.0, return_stylized_content=False):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        adain_style_feat = self.encode_for_inr_decoder_128(style)
        content_feat = self.encode_for_inr_decoder_128(content)
        recon_im = inr_decoder(content_feat)
        loss_recon = self.calc_content_loss(recon_im, content.half())

        # stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = adain(content_feat, adain_style_feat)
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        # stylized_content_feat = whitening(stylized_content_feat)

        stylized_content = inr_decoder(stylized_content_feat).type(torch.float32)
        stylized_content_feat_encoded = self.encode_for_inr_decoder_128(stylized_content)

        stylized_content_feat_encoded_style_loss = self.encode_with_intermediate(stylized_content)

        loss_c = self.calc_content_loss(stylized_content_feat_encoded, stylized_content_feat)
        loss_s = self.calc_style_loss(stylized_content_feat_encoded_style_loss[0], style_feats[0])
        
        for i in range(1, 2):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded_style_loss[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content, loss_recon
        else:
            return loss_c, loss_s
        
    def inr_wct(self, content, style, inr_decoder, alpha=1.0, return_stylized_content=False):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        adain_style_feat = self.encode_for_inr_decoder_64(style)
        content_feat = self.encode_for_inr_decoder_64(content)

        loss_c = 0
        loss_s = 0

        # stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = feature_wct(content_feat, adain_style_feat)
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        # stylized_content_feat = whitening(stylized_content_feat)

        stylized_content = inr_decoder(stylized_content_feat).type(torch.float32)
        stylized_content_feat_encoded = self.encode_for_inr_decoder_64(stylized_content)

        stylized_content_feat_encoded_style_loss = self.encode_with_intermediate(stylized_content)

        # loss_c = self.calc_content_loss(stylized_content_feat_encoded, stylized_content_feat)
        # loss_s = self.calc_style_loss_wct(stylized_content_feat_encoded_style_loss[0], style_feats[0])
        
        
        
        # for i in range(1, 4):
        #     loss_s += self.calc_style_loss_wct(stylized_content_feat_encoded_style_loss[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content
        else:
            return loss_c, loss_s

    def inr_wct(self, content, style, inr_decoder, alpha=1.0, return_stylized_content=False):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        adain_style_feat = self.encode_for_inr_decoder_64(style)
        content_feat = self.encode_for_inr_decoder_64(content)

        loss_c = 0
        loss_s = 0

        # stylized_content_feat = adain(content_feat, style_feats[-1])
        stylized_content_feat = feature_wct(content_feat, adain_style_feat)
        stylized_content_feat = alpha * stylized_content_feat + (1 - alpha) * content_feat

        # stylized_content_feat = whitening(stylized_content_feat)

        stylized_content = inr_decoder(stylized_content_feat).type(torch.float32)
        stylized_content_feat_encoded = self.encode_for_inr_decoder_64(stylized_content)

        stylized_content_feat_encoded_style_loss = self.encode_with_intermediate(stylized_content)

        # loss_c = self.calc_content_loss(stylized_content_feat_encoded, stylized_content_feat)
        # loss_s = self.calc_style_loss_wct(stylized_content_feat_encoded_style_loss[0], style_feats[0])
        
        
        
        # for i in range(1, 4):
        #     loss_s += self.calc_style_loss_wct(stylized_content_feat_encoded_style_loss[i], style_feats[i])
        if return_stylized_content:
            return loss_c, loss_s, stylized_content
        else:
            return loss_c, loss_s

    def get_style_loss(self, content, style, inter_image):
        inter_image = self.upsample(inter_image)

        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)

        stylized_content_feat_encoded = self.encode_with_intermediate(inter_image)
        loss_s = self.calc_style_loss(stylized_content_feat_encoded[0], style_feats[0])

        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_content_feat_encoded[i], style_feats[i])

        return loss_s

