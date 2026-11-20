"""FGSM evasion attack on the CIFAR-10 CNN, with IBM's Adversarial Robustness Toolbox (ART).

Evasion = the attacker changes the INPUT at test time. The model and its training data stay clean.
FGSM (Fast Gradient Sign Method, Goodfellow et al. 2014) takes one step of size eps in the
direction that raises the loss most per pixel:

    x_adv = clip(x + eps * sign(grad_x loss(model(x), y)), 0, 1)

eps is the largest change of any one pixel. 8/255 is the usual CIFAR-10 budget: a change of
8 out of 255 brightness levels, hard to see by eye.

Run:  python art_lab/fgsm_evasion.py            (trains the clean CNN, about 10 epochs)
      python art_lab/fgsm_evasion.py --epochs 3 (faster, a weaker model)
"""

import numpy as np

import common


def main():
    parser = common.base_args(__doc__.splitlines()[0])
    parser.add_argument("--eval", type=int, default=2000, help="test images to attack")
    args = parser.parse_args()

    from art.attacks.evasion import FastGradientMethod

    print(common.versions())
    x_train, y_train, x_test, y_test = common.load_cifar10(args.train, args.test)
    model = common.clean_model(x_train, y_train, args.epochs, args.retrain)
    classifier = common.art_classifier(model)

    x_eval, y_eval = x_test[: args.eval], y_test[: args.eval]
    clean_acc, clean_pred = common.accuracy(classifier, x_eval, y_eval)

    print()
    print(f"FGSM evasion on {len(x_eval)} CIFAR-10 test images")
    print(f"{'eps':>8} {'accuracy':>9} {'flipped':>8} {'max |dx|':>9}")
    print(f"{'clean':>8} {clean_acc:9.1%} {'-':>8} {'-':>9}")
    attacked = {}
    for eps_255 in (1, 2, 4, 8, 16):
        eps = eps_255 / 255.0
        attack = FastGradientMethod(estimator=classifier, eps=eps, batch_size=256)
        x_adv = attack.generate(x=x_eval)
        adv_acc, adv_pred = common.accuracy(classifier, x_adv, y_eval)
        # "flipped" = images the model got right before the attack and wrong after it.
        flipped = np.mean((clean_pred == y_eval) & (adv_pred != y_eval))
        max_dx = float(np.max(np.abs(x_adv - x_eval)))
        attacked[eps_255] = adv_acc
        print(f"{eps_255:>4}/255 {adv_acc:9.1%} {flipped:8.1%} {max_dx * 255:7.1f}/255")

    print()
    print(f"RESULT fgsm clean_accuracy={clean_acc:.4f} attacked_accuracy_eps8={attacked[8]:.4f}")


if __name__ == "__main__":
    main()
