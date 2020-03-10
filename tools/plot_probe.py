import torch
import numpy as np
import matplotlib.pyplot as plt


# Load tensors
#savepath = 'probe/probe_loss_out_val.pt'
#savepath = 'probe/probe_fpn_val.pt'
savepath = 'probe/probe_fpn_train50000.pt'
#gtsavepath = 'probe/gt_boxes_val.pt'
gtsavepath = 'probe/gt_boxes_train50000.pt'
total = torch.load(savepath, map_location='cpu')
gts = torch.load(gtsavepath, map_location='cpu')
print("total:", total, total.shape)
print("gts:", gts.shape)

# Set histogram/bar bins
max_sqrt_area = 1030
sqrt_area_bins = list(range(0, max_sqrt_area, 5))

# Compute sqrt_areas = sqrt(widths * heights)
sqrt_areas = torch.sqrt((total[:, 2] - total[:, 0]) * (total[:, 3] - total[:, 1]))
#print("sqrt_areas:", sqrt_areas, sqrt_areas.min(), sqrt_areas.max(), sqrt_areas.shape)
sqrt_areas_gt = torch.sqrt((gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1]))
#print("sqrt_areas_gt:", sqrt_areas_gt, sqrt_areas_gt.min(), sqrt_areas_gt.max(), sqrt_areas_gt.shape)


# Plot instance count histograms
fig, axs = plt.subplots(2, 1)
axs[0].hist(sqrt_areas_gt, sqrt_area_bins)
axs[0].set_title('All GT Targets')
axs[0].set_xlim(left=0, right=max_sqrt_area)
axs[0].set_ylabel('Instances')

axs[1].hist(sqrt_areas, sqrt_area_bins)
axs[1].set_title('Matched Anchor Targets')
axs[1].set_xlim(left=0, right=max_sqrt_area)
axs[1].set_ylabel('Instances')
axs[1].set_xlabel('sqrt(Area)')

fig.tight_layout()
plt.show()


# ### Accumulate losses over attributes
# bins = sqrt_area_bins
# accumulated_losses_sum = []
# accumulated_losses_avg = []
# losses = total[:, -2:]
# #print("losses:", losses.shape)
# #print("sqrt_areas:", sqrt_areas.shape)
# sum_losses_sum = 0
# sum_losses_avg = 0
#
# for i in range(len(bins) - 1):
#     # Make binary mask for current bin
#     # Note: element-wise multiplication of binary tensors is == element-wise 'and'
#     mask = (sqrt_areas >= bins[i]) * (sqrt_areas < bins[i+1])
#     num_boxes = mask.sum().item()
#     # Slice out losses from boxes that are in this bin
#     curr_losses = losses[mask.nonzero()].squeeze()
#     # Accumulate losses
#     curr_acc_loss_sum = curr_losses.sum().item()
#     curr_acc_loss_avg = curr_acc_loss_sum / num_boxes if num_boxes > 0 else 0
#     # Append current accumulated loss to accumulated_losses list
#     accumulated_losses_sum.append(curr_acc_loss_sum)
#     accumulated_losses_avg.append(curr_acc_loss_avg)
#     # Add to sums
#     sum_losses_sum += curr_acc_loss_sum
#     sum_losses_avg += curr_acc_loss_avg
#
# # Make accumulated_losses a torch tensor
# accumulated_losses_sum = torch.tensor(accumulated_losses_sum)
# accumulated_losses_avg = torch.tensor(accumulated_losses_avg)
# # Normalize accumulated_losses so that they sum to 1
# accumulated_losses_sum /= sum_losses_sum
# accumulated_losses_avg /= sum_losses_avg
#
#
# ### Plot bar chart
# # Create x_ticks and tick labels
# x = torch.arange(len(accumulated_losses_sum))
# x_ticks = torch.arange(0, len(bins), step=20)
# bins_t = torch.tensor(bins)
# bin_labels = bins_t[1:][x_ticks].tolist()
# bin_labels = [str(b) for b in bin_labels]
#
# fig, axs = plt.subplots(2, 1)
# axs[0].bar(x, accumulated_losses_sum, width=1.0)
# axs[0].set_title('Total Loss Contribution')
# axs[0].set_xticks(x_ticks, bin_labels)
# axs[0].set_xlim(x[0], x[-1])
# axs[0].set_ylabel('% loss')
#
# axs[1].bar(x, accumulated_losses_avg, width=1.0)
# axs[1].set_title('Average Loss Contribution')
# axs[1].set_xticks(x_ticks, bin_labels)
# axs[1].set_xlim(x[0], x[-1])
# axs[1].set_xlabel('sqrt(Area)')
# axs[1].set_ylabel('% loss')
# fig.tight_layout()
# plt.show()
