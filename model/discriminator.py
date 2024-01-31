import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, seq_length, latent_dim):
        super(Discriminator, self).__init__()
        # Define basic parameters
        self.seq_length = seq_length
        self.lent_dim = latent_dim

        # Define network structure components
        self.linear1 = nn.Linear(in_features=self.seq_length * self.lent_dim, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=1)
        self.flatten = nn.Flatten(start_dim=2)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, seqs):
        seqs = self.flatten(seqs)
        output = self.linear1(seqs)
        output = self.leaky_relu(output)
        output = self.linear2(output)
        # print(output.shape)
        return output

# def discriminator_fully_connected(x, labels, df_dim, number_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
#                                   pooling='avg', update_collection=None, act=nn.ReLU, scope_name='Discriminator',
#                                   reuse=False):
        # x = tf.layers.flatten(x)
        # x = tf.layers.dense(x, df_dim, name="dense_1")
        # x = leaky_relu(x)
        # tf.summary.histogram(x.name, x)
        # output = tf.layers.dense(x, 1, name="dense_2")
        # tf.summary.histogram(output.name, output)
        # return output
    


# def original_discriminator(x, labels, df_dim, number_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
#                            pooling='avg', update_collection=None, act=tf.nn.relu, scope_name='Discriminator',
#                            reuse=False):
#         conv1 = nn.Conv2d(
#             inputs=x,
#             filters=df_dim / 4,
#             kernel_size=[3, 3],
#             strides=(2, 2),
#             padding="same",
#             activation=leaky_relu,
#             name="dconv1")
#         tf.summary.histogram(conv1.name, conv1)
#         # Convolutional Layer #2
#         conv2 = tf.layers.conv2d(
#             inputs=conv1,
#             filters=df_dim / 2,
#             kernel_size=[3, 3],
#             strides=(2, 2),
#             padding="same",
#             activation=leaky_relu,
#             name="dconv2")
#         tf.summary.histogram(conv2.name, conv2)
#         conv3 = tf.layers.conv2d(
#             inputs=conv2,
#             filters=df_dim,
#             kernel_size=[3, 3],
#             strides=(2, 2),
#             padding="same",
#             activation=leaky_relu,
#             name="dconv3")
#         tf.summary.histogram(conv3.name, conv3)
#         flat = tf.layers.flatten(conv3, name="dflat")
#         output = tf.layers.dense(inputs=flat,
#                                  activation=None,
#                                  units=1,
#                                  name="doutput")
#         output = tf.reshape(output, [-1])
#         tf.summary.histogram(output.name, output)
#         return output
# 
# 
# def discriminator_resnet(x, labels, df_dim, number_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
#                          pooling='avg', update_collection=None, act=tf.nn.relu, scope_name='Discriminator',
#                          reuse=False):
#     with tf.variable_scope(scope_name) as scope:
#         if reuse:
#             scope.reuse_variables()
#         h0 = block(x, df_dim, 'd_optimized_block1', act=act)  # 12 * 12
#         h1 = block(h0, df_dim * 2, 'd_block2', act=act)  # 6 * 6
#         h2 = block(h1, df_dim * 4, 'd_block3', act=act)  # 3 * 3
#         tf.summary.histogram(h2.name, h2)
#         # h3 = block(h2, df_dim * 4, 'd_block4', update_collection, act=act)  # 8 * 8 # 3*12
#         # h4 = block(h3, df_dim * 8, 'd_block5', update_collection, act=act)  # 3*6
#         h5 = block(h2, df_dim * 8, 'd_block6', False, act=act)
#         h5_act = act(h5)
#         tf.summary.histogram(h5_act.name, h5_act)
#         h6 = tf.reduce_sum(h5_act, [1, 2])
#         output = ops.linear(h6, 1, scope='d_linear')
#         tf.summary.histogram(output.name, output)
#         return output


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
