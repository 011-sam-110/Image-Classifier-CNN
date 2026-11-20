"""Backdoor poisoning attack on the CIFAR-10 CNN, with IBM's Adversarial Robustness Toolbox (ART).

Poisoning = the attacker changes the TRAINING DATA. A backdoor (BadNets, Gu et al. 2017) is one
kind: the attacker stamps a small trigger on a few training images and relabels them as one
target class. The model learns two rules. On clean images it works as normal. On any image with
the trigger it answers the target class. A test on clean data alone cannot see the backdoor.

Here the trigger is a 3 x 3 black and white checkerboard in the bottom-right corner, and the
target class is "Plane". ART's PoisoningAttackBackdoor applies it.

Run:  python art_lab/backdoor_poisoning.py              (poisons 5% of the training set)
      python art_lab/backdoor_poisoning.py --poison 0.01 (a smaller dose)
The script reuses the clean model from fgsm_evasion.py when art_lab/out/ holds one.
"""

import numpy as np

import common

TARGET = 0  # "Plane"


def add_checkerboard(x):
    """Stamp a 3 x 3 checkerboard (white, black, white, ...) one pixel in from the corner.

    x: images of shape (N, 32, 32, 3), values in [0, 1]. Returns a copy.
    """
    x = np.copy(x)
    for row in range(3):
        for col in range(3):
            x[:, -4 + row, -4 + col, :] = 1.0 if (row + col) % 2 == 0 else 0.0
    return x


def main():
    parser = common.base_args(__doc__.splitlines()[0])
    parser.add_argument("--poison", type=float, default=0.05, help="share of the training set to poison")
    args = parser.parse_args()

    from art.attacks.poisoning import PoisoningAttackBackdoor

    print(common.versions())
    x_train, y_train, x_test, y_test = common.load_cifar10(args.train, args.test)

    # 1. Poison: pick training images that are NOT planes, add the trigger, relabel them "Plane".
    rng = np.random.default_rng(common.SEED)
    candidates = np.flatnonzero(y_train != TARGET)
    n_poison = int(args.poison * len(x_train))
    picked = rng.choice(candidates, size=n_poison, replace=False)
    backdoor = PoisoningAttackBackdoor(perturbation=add_checkerboard)
    x_poison, y_poison = backdoor.poison(x_train[picked], y=np.array([TARGET]), broadcast=True)
    x_bad, y_bad = np.copy(x_train), np.copy(y_train)
    x_bad[picked] = x_poison
    y_bad[picked] = np.asarray(y_poison).reshape(-1).astype(y_train.dtype)
    changed = np.mean(np.any(x_bad[picked] != x_train[picked], axis=(1, 2, 3)))
    print(f"[poison] {n_poison} of {len(x_train)} training images ({args.poison:.0%}) now carry the "
          f"trigger and the label {common.CLASS_NAMES[TARGET]} (trigger present on {changed:.0%} of them)")

    # 2. Train one model on the poisoned set, and get the clean model as a control.
    common.set_seed()
    poisoned = common.art_classifier(common.train(common.build_cnn(), x_bad, y_bad, args.epochs, "poisoned"))
    clean = common.art_classifier(common.clean_model(x_train, y_train, args.epochs, args.retrain))

    # 3. Test: clean test images, and the same non-plane test images with the trigger.
    keep = y_test != TARGET
    x_trig, y_trig = add_checkerboard(x_test[keep]), y_test[keep]

    rows = []
    for name, classifier in (("clean model", clean), ("poisoned model", poisoned)):
        clean_acc, _ = common.accuracy(classifier, x_test, y_test)
        trig_acc, trig_pred = common.accuracy(classifier, x_trig, y_trig)
        success = float(np.mean(trig_pred == TARGET))
        rows.append((name, clean_acc, trig_acc, success))

    print()
    print(f"Backdoor poisoning: {len(x_test)} clean test images, {len(x_trig)} triggered non-plane test images")
    print(f"{'model':<15} {'clean acc':>10} {'triggered acc':>14} {'says Plane':>11}")
    for name, clean_acc, trig_acc, success in rows:
        print(f"{name:<15} {clean_acc:10.1%} {trig_acc:14.1%} {success:11.1%}")

    _, p_clean, p_trig, p_success = rows[1]
    print()
    print(f"RESULT backdoor poison={args.poison} clean_accuracy={p_clean:.4f} "
          f"attacked_accuracy={p_trig:.4f} attack_success={p_success:.4f}")


if __name__ == "__main__":
    main()
