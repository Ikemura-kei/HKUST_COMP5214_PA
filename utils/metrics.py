
def get_accuracy(gt, pred):
    '''
    gt: a torch tensor of shape (B), of all ground truth labels
    pred: a torch tensor of shape (B), of all predicted labels
    '''
    
    correct = gt[gt==pred]
    acc = correct.size(0) / gt.size(0)
    
    return acc