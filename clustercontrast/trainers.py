from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch


# class ClusterContrastTrainer_pretrain(object):
#     def __init__(self, encoder, memory=None):
#         super(ClusterContrastTrainer_pretrain, self).__init__()
#         self.encoder = encoder
#         self.memory_ir = memory
#         self.memory_rgb = memory
#     def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400):
#         self.encoder.train()
#
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#
#         losses = AverageMeter()
#
#         end = time.time()
#         for i in range(train_iters):
#             # load data
#             inputs_ir = data_loader_ir.next()
#             inputs_rgb = data_loader_rgb.next()
#             data_time.update(time.time() - end)
#
#             # process inputs
#             inputs_ir, labels_ir, indexes_ir = self._parse_data(inputs_ir)
#             inputs_rgb, labels_rgb, indexes_rgb = self._parse_data(inputs_rgb)
#             # forward
#             _,f_out_rgb,f_out_ir,labels_rgb,labels_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
#             loss_ir = self.memory_ir(f_out_ir, labels_ir)
#             loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
#             loss = loss_ir+loss_rgb
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             losses.update(loss.item())
#
#             # print log
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             if (i + 1) % print_freq == 0:
#                 print('Epoch: [{}][{}/{}]\t'
#                       'Time {:.3f} ({:.3f})\t'
#                       'Data {:.3f} ({:.3f})\t'
#                       'Loss {:.3f} ({:.3f})\t'
#                       'Loss ir {:.3f}\t'
#                       'Loss rgb {:.3f}\t'
#                       .format(epoch, i + 1, len(data_loader_rgb),
#                               batch_time.val, batch_time.avg,
#                               data_time.val, data_time.avg,
#                               losses.val, losses.avg,loss_ir,loss_rgb))
#
#     def _parse_data(self, inputs):
#         imgs, _, pids, _, indexes = inputs
#         return imgs.cuda(), pids.cuda(), indexes.cuda()
#
#     def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
#         return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


# class ClusterContrastTrainer_pretrain(object):
#     def __init__(self, encoder,memory=None):
#         super(ClusterContrastTrainer_pretrain, self).__init__()
#         self.encoder = encoder
#         self.memory_ir = memory
#         self.memory_rgb = memory
#
#     def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400):
#         self.encoder.train()
#
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#
#         losses = AverageMeter()
#
#         end = time.time()
#         for i in range(train_iters):
#             # load data
#             inputs_ir = data_loader_ir.next()
#             inputs_rgb = data_loader_rgb.next()
#             data_time.update(time.time() - end)
#
#             # process inputs
#             inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
#             inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
#             # forward
#             inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
#             labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
#             _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
#
#             loss_ir = self.memory_ir(f_out_ir, labels_ir)
#             loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
#             loss = loss_ir + loss_rgb
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             losses.update(loss.item())
#
#             # print log
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             if (i + 1) % print_freq == 0:
#                 print('Epoch: [{}][{}/{}]\t'
#                       'Time {:.3f} ({:.3f})\t'
#                       'Data {:.3f} ({:.3f})\t'
#                       'Loss {:.3f} ({:.3f})\t'
#                       'Loss ir {:.3f}\t'
#                       'Loss rgb {:.3f}\t'
#                       .format(epoch, i + 1, len(data_loader_rgb),
#                               batch_time.val, batch_time.avg,
#                               data_time.val, data_time.avg,
#                               losses.val, losses.avg,loss_ir,loss_rgb))
#
#     def _parse_data_rgb(self, inputs):
#         imgs,imgs1, _, pids, _, indexes = inputs
#         return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()
#
#     def _parse_data_ir(self, inputs):
#         imgs, _, pids, _, indexes = inputs
#         return imgs.cuda(), pids.cuda(), indexes.cuda()
#
#     def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
#         return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)

# class ClusterContrastTrainer_joint(object):
#     def __init__(self, encoder,memory=None):
#         super(ClusterContrastTrainer_joint, self).__init__()
#         self.encoder = encoder
#         self.memory_ir = memory
#         self.memory_rgb = memory
#         self.memory_all = memory
#     def train(self, epoch, data_loader_ir,data_loader_rgb, data_loader_all_ir, data_loader_all_rgb, optimizer, print_freq=10, train_iters=400):
#         self.encoder.train()
#
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#         losses = AverageMeter()
#         end = time.time()
#
#         for i in range(train_iters):
#             # load data
#             inputs_ir = data_loader_ir.next()
#             inputs_rgb = data_loader_rgb.next()
#             data_time.update(time.time() - end)
#
#             # process inputs
#             inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
#             inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
#             # forward
#             inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
#             labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
#
#             _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
#
#             loss_ir = self.memory_ir(f_out_ir, labels_ir)
#             loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
#             loss = loss_ir + loss_rgb
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             losses.update(loss.item())
#
#             inputs_all_ir = data_loader_all_ir.next()
#             inputs_all_rgb = data_loader_all_rgb.next()
#             # process inputs
#             inputs_all_ir, labels_all_ir, indexes_all_ir = self._parse_data_ir(inputs_all_ir)
#             inputs_all_rgb, inputs_all_rgb1, labels_all_rgb, indexes_all_rgb = self._parse_data_rgb(inputs_all_rgb)
#             # forward
#             inputs_all_rgb = torch.cat((inputs_all_rgb, inputs_all_rgb1), 0)
#             labels_all_rgb = torch.cat((labels_all_rgb, labels_all_rgb), -1)
#
#             _, f_out_all_rgb, f_out_all_ir, labels_all_rgb, labels_all_ir, pool_all_rgb, pool_all_ir = self._forward(inputs_all_rgb, inputs_all_ir,
#                                                                                                 label_1=labels_all_rgb,
#                                                                                                 label_2=labels_all_ir, modal=0)
#             loss_all_ir = self.memory_all(f_out_all_ir, labels_all_ir)
#             loss_all_rgb = self.memory_all(f_out_all_rgb, labels_all_rgb)
#
#             loss2 = loss_all_ir + loss_all_rgb
#
#             optimizer.zero_grad()
#             loss2.backward()
#             optimizer.step()
#
#             # print log
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             if (i + 1) % print_freq == 0:
#                 print('Epoch: [{}][{}/{}]\t'
#                       'Time {:.3f} ({:.3f})\t'
#                       'Data {:.3f} ({:.3f})\t'
#                       'Loss {:.3f} ({:.3f})\t'
#                       'Loss ir {:.3f}\t'
#                       'Loss rgb {:.3f}\t'
#                       'Loss ir all {:.3f}\t'
#                       'Loss rgb all {:.3f}\t'
#                       'Loss all {:.3f}\t'
#                       .format(epoch, i + 1, len(data_loader_rgb),
#                               batch_time.val, batch_time.avg,
#                               data_time.val, data_time.avg,
#                               losses.val, losses.avg,loss_ir,loss_rgb,loss_all_ir,loss_all_rgb,loss2))
#
#     def _parse_data_rgb(self, inputs):
#         imgs,imgs1, _, pids, _, indexes = inputs
#         return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()
#
#     def _parse_data_ir(self, inputs):
#         imgs, _, pids, _, indexes = inputs
#         return imgs.cuda(), pids.cuda(), indexes.cuda()
#
#     def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
#         return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)

class ClusterContrastTrainerPretrain:
    """
    Trainer class for pre-training with cluster contrastive learning.

    Attributes:
        encoder (nn.Module): The encoder network.
        memory_ir (Memory): Memory module for infrared modality.
        memory_rgb (Memory): Memory module for RGB modality.
    """

    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainerPretrain, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory

    def train(self, epoch, data_loader_ir, data_loader_rgb, optimizer, print_freq=10, train_iters=400):
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.
            data_loader_ir (DataLoader): DataLoader for infrared modality.
            data_loader_rgb (DataLoader): DataLoader for RGB modality.
            optimizer (Optimizer): Optimizer for training.
            print_freq (int): Frequency of printing log.
            train_iters (int): Number of iterations per epoch.
        """
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i in range(train_iters):
            # Load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # Process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)

            # Concatenate RGB inputs and labels
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)

            # Forward pass
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = self._forward(inputs_rgb, inputs_ir,
                                                                                             label_1=labels_rgb,
                                                                                             label_2=labels_ir, modal=0)

            # Compute loss
            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir + loss_rgb

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # Print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss IR {:.3f}\t'
                      'Loss RGB {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir, loss_rgb))

    def _parse_data_rgb(self, inputs):
        """
        Parse RGB data inputs.

        Args:
            inputs (tuple): Input data.

        Returns:
            Tuple: Processed images, labels, and indexes for RGB modality.
        """
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        """
        Parse infrared data inputs.

        Args:
            inputs (tuple): Input data.

        Returns:
            Tuple: Processed images, labels, and indexes for infrared modality.
        """
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None, label_2=None, modal=0):
        """
        Forward pass through the encoder.

        Args:
            x1 (Tensor): Input tensor 1.
            x2 (Tensor): Input tensor 2.
            label_1 (Tensor): Labels for input 1.
            label_2 (Tensor): Labels for input 2.
            modal (int): Modality indicator.

        Returns:
            Tuple: Encoder output.
        """
        return self.encoder(x1, x2, modal=modal, label_1=label_1, label_2=label_2)


class ClusterContrastTrainerJoint:
    """
    Trainer class for cluster contrast training with inter modality inputs.

    Attributes:
        encoder (nn.Module): The encoder network.
        memory_ir (Memory): Memory module for infrared modality.
        memory_rgb (Memory): Memory module for RGB modality.
        memory_all (Memory): Memory module for combined modalities.
    """

    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainerJoint, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_all = memory

    def train(self, epoch, data_loader_ir, data_loader_rgb, data_loader_all_ir, data_loader_all_rgb, optimizer,
              print_freq=10, train_iters=400):
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.
            data_loader_ir (DataLoader): DataLoader for infrared modality.
            data_loader_rgb (DataLoader): DataLoader for RGB modality.
            data_loader_all_ir (DataLoader): DataLoader for combined modalities (infrared).
            data_loader_all_rgb (DataLoader): DataLoader for combined modalities (RGB).
            optimizer (Optimizer): Optimizer for training.
            print_freq (int): Frequency of printing log.
            train_iters (int): Number of iterations per epoch.
        """
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i in range(train_iters):
            # Load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # Process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)

            # Concatenate RGB inputs and labels
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)

            # Forward pass
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = self._forward(inputs_rgb, inputs_ir,
                                                                                             label_1=labels_rgb,
                                                                                             label_2=labels_ir, modal=0)
            # Compute loss
            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir + loss_rgb

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # Load data for combined modalities
            inputs_all_ir = data_loader_all_ir.next()
            inputs_all_rgb = data_loader_all_rgb.next()

            # Process combined inputs
            inputs_all_ir, labels_all_ir, indexes_all_ir = self._parse_data_ir(inputs_all_ir)
            inputs_all_rgb, inputs_all_rgb1, labels_all_rgb, indexes_all_rgb = self._parse_data_rgb(inputs_all_rgb)

            # Concatenate all RGB inputs and labels
            inputs_all_rgb = torch.cat((inputs_all_rgb, inputs_all_rgb1), 0)
            labels_all_rgb = torch.cat((labels_all_rgb, labels_all_rgb), -1)

            # Forward pass for combined modalities
            _, f_out_all_rgb, f_out_all_ir, labels_all_rgb, labels_all_ir, pool_all_rgb, pool_all_ir = self._forward(
                inputs_all_rgb, inputs_all_ir, label_1=labels_all_rgb, label_2=labels_all_ir, modal=0)

            # Compute loss for combined modalities
            loss_all_ir = self.memory_all(f_out_all_ir, labels_all_ir)
            loss_all_rgb = self.memory_all(f_out_all_rgb, labels_all_rgb)
            loss2 = loss_all_ir + loss_all_rgb

            # Backpropagation and optimization for combined modalities
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()

            # Print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss IR {:.3f}\t'
                      'Loss RGB {:.3f}\t'
                      'Loss IR All {:.3f}\t'
                      'Loss RGB All {:.3f}\t'
                      'Loss All {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir, loss_rgb, loss_all_ir, loss_all_rgb, loss2))

    def _parse_data_rgb(self, inputs):
        """
        Parse RGB data inputs.

        Args:
            inputs (tuple): Input data.

        Returns:
            Tuple: Processed images, labels, and indexes for RGB modality.
        """
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        """
        Parse infrared data inputs.

        Args:
            inputs (tuple): Input data.

        Returns:
            Tuple: Processed images, labels, and indexes for infrared modality.
        """
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None, label_2=None, modal=0):
        """
        Forward pass through the encoder.

        Args:
            x1 (Tensor): Input tensor 1.
            x2 (Tensor): Input tensor 2.
            label_1 (Tensor): Labels for input 1.
            label_2 (Tensor): Labels for input 2.
            modal (int): Modality indicator.

        Returns:
            Tuple: Encoder output.
        """
        return self.encoder(x1, x2, modal=modal, label_1=label_1, label_2=label_2)


