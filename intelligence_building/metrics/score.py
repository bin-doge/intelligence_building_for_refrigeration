import numpy as np

__all__ = ['out_of_distribution_score']
def out_of_distribution_score(y_pred,y_true,mask):
    """
    compute the out_of_distribution_score
    mask: Array-like object, 1 indicates accept the predict, 0 indicates reject the predict

                        right        wrong
                    ————————————————————————————
                    |            |            |
          accept    |     ar     |     aw     |
                    ———————————————————————————
                    |            |            |
          reject    |     rr     |     rw     |
                    ———————————————————————————

        
         p(handle_well) = (ar+rw)/(ar+aw+rr+rw)
         p(right|accept) = ar/(ar+aw)
    """
    accept_index = (mask==1)
    reject_index = ~accept_index
    ar = np.sum(y_pred[accept_index]==y_true[accept_index])
    aw = len(y_true[accept_index])-ar
    rr = np.sum(y_pred[reject_index]==y_true[reject_index])
    rw = len(y_pred[reject_index])-rr
    return 1.0*(ar+rw)/(ar+aw+rr+rw),1.0*ar/(ar+aw)


