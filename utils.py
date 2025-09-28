# utils.py
import torch
import torch.nn.functional as F

def freeze_encoder(model):
    """Freeze the encoder layers of a transformer model."""
    for _, param in model.model.encoder.named_parameters():
        param.requires_grad = False


def print_and_flush(text):
    """Print text and flush immediately."""
    print(text, end='', flush=True)


# Stopword list for entity masking
IGNORE_WORDS = {
    'the','a','in','is','for','of','and','to','with','without','on','at',
    'by','under','over','between','into','through','during','before','after',
    'above','below','from','up','down','out','off','again','further','then','once',
    'here','there','when','where','why','how','all','any','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don',"don't",'should',"should've",'now',
    'd','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",'didn',
    "didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",
    'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",
    'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't',
    'won',"won't",'wouldn',"wouldn't","new"
}


def is_in_answers(entity: str, answers) -> bool:
    """Return True if filtered entity words appear in the answers list."""
    filtered_words = [w for w in entity.lower().split() if w not in IGNORE_WORDS]
    return any(word in answers for word in filtered_words)


def mask_entity(entity_spans, answers, desired_size=80):
    """Create an entity mask padded to desired_size."""
    x = torch.tensor([1 if is_in_answers(e, answers) else 0 for e in entity_spans])
    padding = (0, desired_size - x.size(0))
    return F.pad(x, pad=padding, mode='constant', value=0)
