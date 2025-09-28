# eval.py
import nltk
from rouge import Rouge

def check_score(preds, refs):
    """Compute BLEU-1..4, METEOR, ROUGE-L"""
    print("#### SAMPLE PREDICTIONS ####", preds[:5])
    print("#### SAMPLE REFERENCES ####", refs[:5])

    rouge = Rouge()
    total_rouge, total_meteor, num = 0.0, 0.0, 0
    golds, pred_tokens = [], []

    for p, g in zip(preds, refs):
        if p.strip():
            total_rouge += rouge.get_scores([p], [g])[0]['rouge-l']['f']
            ptok, gtok = nltk.word_tokenize(p), nltk.word_tokenize(g)
            pred_tokens.append(ptok)
            golds.append([gtok])
            total_meteor += nltk.translate.meteor_score.meteor_score([gtok], ptok)
            num += 1

    bleu1 = nltk.translate.bleu_score.corpus_bleu(golds, pred_tokens, weights=(1,0,0,0))
    bleu2 = nltk.translate.bleu_score.corpus_bleu(golds, pred_tokens, weights=(0.5,0.5,0,0))
    bleu3 = nltk.translate.bleu_score.corpus_bleu(golds, pred_tokens, weights=(0.33,0.33,0.33,0))
    bleu4 = nltk.translate.bleu_score.corpus_bleu(golds, pred_tokens, weights=(0.25,0.25,0.25,0.25))
    meteor = total_meteor / max(1, num)
    rouge_l = total_rouge / max(1, num)

    print(f"BLEU1: {bleu1:.4f}  BLEU2: {bleu2:.4f}  BLEU3: {bleu3:.4f}  BLEU4: {bleu4:.4f}")
    print(f"METEOR: {meteor:.4f}  ROUGE-L: {rouge_l:.4f}")

    return (bleu1 + meteor + rouge_l) / 3
