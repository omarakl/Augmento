# Contributing to Augmento

Thanks for your interest. Small, focused contributions are the easiest to review.

## Report a bug

Open an issue and include:
- What you did and what you expected
- What happened instead, with any error text from the log box in the app
- Your operating system and Python version
- An example image or folder layout if it is relevant

## Suggest a feature

Open an issue that describes the problem you want to solve, not just the solution. Examples of useful ideas: new augmentations, support for COCO or Pascal VOC labels, and new resize modes.

## Send a code change

1. Fork the repository and create a branch.
2. Install the requirements: `pip install -r requirements.txt`
3. Make your change. To add an augmentation, write the function and add one row to `REGISTRY` in `augmento.py`.
4. Add or update tests in `test_augmento.py`.
5. Run `python test_augmento.py` and make sure every test passes.
6. Open a pull request that explains what changed and why.

## Style

- Keep functions small and name them clearly.
- Keep the user interface responsive: long work belongs in a worker function that talks to the window through the queue.
- Never overwrite or delete a user's files without a confirmation.
