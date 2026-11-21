"""Restore the inference stage actually used by a saved Student checkpoint."""
def checkpoint_warm(state):
    epoch=int(state['epoch_next'])-int(bool(state.get('resume_at_epoch_boundary',False)))
    if epoch<0:raise ValueError('Invalid checkpoint epoch metadata')
    return epoch==0
