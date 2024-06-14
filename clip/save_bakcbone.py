import torch
import clip

model, preprocess = clip.load("RN101", device='cpu')
visual = model.visual

torch.save(visual.state_dict(), 'clip-res101.pth')