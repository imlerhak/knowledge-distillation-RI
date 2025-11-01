import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, targets, alpha=0.9, temperature=4.0, use_kd=True):
    """
    Hinton KD: CE(y, s) + (1-alpha) * KL( softmax(t/T) || softmax(s/T) ) * T^2
    use_kd=False -> samo CE (za ESKD fazu kada isključimo KD u kasnijim epohama).
    """
    ce = F.cross_entropy(student_logits, targets)
    if not use_kd:
        return ce
    T = temperature
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t    = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(log_p_s, p_t, reduction='batchmaean') * (T * T)
    return alpha * ce + (1.0 - alpha) * kd
