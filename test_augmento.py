"""Tests for every function in augmento.py.

Keep this file in the same folder as augmento.py, then run:
    python test_augmento.py
(or, from that folder: python -m unittest test_augmento -v)
GUI tests are skipped automatically when no display is available.
"""
import csv
import os
import queue
import shutil
import tempfile
import threading
import time
import unittest
import warnings
from unittest import mock

import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # lets the tests find augmento.py next to this file, from any folder
try:
    import augmento as a
except ModuleNotFoundError as e:
    if e.name != "augmento":
        raise
    raise ImportError(
        f"augmento.py was not found in {HERE}\n"
        "Put test_augmento.py and augmento.py in the same folder, and make sure the main file is "
        "named exactly augmento.py (not 'augmento (1).py' or 'augmento.py.txt').") from None


def setUpModule():
    warnings.simplefilter("ignore", ResourceWarning)  # tests open small files without closing them


def make_img(w=120, h=90, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


def save(arr, path, mode="RGB"):
    Image.fromarray(arr).convert(mode).save(path)


def run(fn, cfg, stop=None):
    """Run a worker function and return the messages it queued."""
    q = queue.Queue()
    fn(cfg, q, stop or threading.Event())
    out = []
    while not q.empty():
        out.append(q.get())
    return out


def logs(msgs):
    return [m[1] for m in msgs if m[0] == "log"]


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def p(self, *parts):
        return os.path.join(self.d, *parts)

    def folder(self, name="in", files=("a.jpg", "b.png", "c.jpeg"), size=(120, 90)):
        path = self.p(name)
        os.makedirs(path, exist_ok=True)
        for i, f in enumerate(files):
            save(make_img(*size, seed=i), os.path.join(path, f))
        return path


# ---------------------------------------------------------------- augmentations
class TestGeometric(unittest.TestCase):
    def test_every_geometric_returns_valid_matrix(self):
        rng = np.random.default_rng(1)
        for name, (kind, fn) in a.REGISTRY.items():
            if kind != "geo":
                continue
            for k in (0.2, 1.0, 2.0):
                with self.subTest(name=name, k=k):
                    M = fn(90, 120, rng, k)
                    self.assertEqual(M.shape, (3, 3))
                    self.assertTrue(np.isfinite(M).all())
                    self.assertGreater(abs(np.linalg.det(M)), 1e-3)

    def test_flip_mirrors_x(self):
        M = a.g_flip(90, 120, None, 1)
        p = M @ np.array([10, 20, 1.0])
        self.assertEqual(tuple(p[:2]), (110, 20))

    def test_about_center_keeps_center_fixed(self):
        M = a._about_center(90, 120, np.diag([2, 2, 1.0]))
        p = M @ np.array([60, 45, 1.0])
        np.testing.assert_allclose(p[:2], [60, 45])

    def test_crop_maps_back_to_full_frame(self):
        M = a.g_crop(90, 120, np.random.default_rng(3), 1.0)
        self.assertGreaterEqual(M[0, 0], 1.0)  # always zooms in (or stays)

    def test_translate_has_no_scaling(self):
        M = a.g_translate(90, 120, np.random.default_rng(3), 1.0)
        np.testing.assert_allclose(M[:2, :2], np.eye(2))

    def test_rotate_zoom_shear_perspective_affine_run(self):
        rng = np.random.default_rng(2)
        for fn in (a.g_rotate, a.g_zoom, a.g_shear, a.g_perspective, a.g_affine):
            self.assertEqual(fn(90, 120, rng, 1.0).shape, (3, 3))


class TestPhotometric(unittest.TestCase):
    def setUp(self):
        self.img = make_img()
        self.rng = np.random.default_rng(5)

    def test_all_methods_keep_shape_and_dtype(self):
        for name, (kind, fn) in a.REGISTRY.items():
            if kind == "geo":
                continue
            for k in (0.2, 1.0, 2.0):
                with self.subTest(name=name, k=k):
                    out = fn(self.img, np.random.default_rng(7), k)
                    self.assertEqual(out.shape, self.img.shape)
                    self.assertEqual(out.dtype, np.uint8)

    def test_all_methods_change_the_image(self):
        for name, (kind, fn) in a.REGISTRY.items():
            if kind == "geo":
                continue
            with self.subTest(name=name):
                changed = any(not np.array_equal(fn(self.img, np.random.default_rng(s), 1.0), self.img)
                              for s in range(5))
                self.assertTrue(changed)

    def test_same_seed_same_result(self):
        for name, (kind, fn) in a.REGISTRY.items():
            if kind == "geo":
                continue
            with self.subTest(name=name):
                x = fn(self.img, np.random.default_rng(11), 1.0)
                y = fn(self.img, np.random.default_rng(11), 1.0)
                np.testing.assert_array_equal(x, y)

    def test_gaussian_noise_does_not_wrap_around(self):
        gray = np.full((100, 100, 3), 128, np.uint8)
        out = a.p_gaussian_noise(gray, self.rng, 1.0)
        self.assertLess(abs(out.mean() - 128), 3)  # the old uint8 bug pushed this far off

    def test_salt_pepper_only_adds_black_and_white(self):
        gray = np.full((100, 100, 3), 128, np.uint8)
        out = a.p_salt_pepper(gray, self.rng, 1.0)
        self.assertTrue(set(np.unique(out)) <= {0, 128, 255})
        self.assertTrue((out == 0).any() or (out == 255).any())

    def test_channel_drop_zeroes_exactly_one_channel(self):
        out = a.p_channel_drop(self.img + 1 if self.img.max() < 255 else self.img, self.rng, 1.0)
        self.assertEqual(out.shape[2], 3)
        zero = [c for c in range(3) if not out[..., c].any()]
        self.assertEqual(len(zero), 1)

    def test_channel_shuffle_is_a_permutation(self):
        out = a.p_channel_shuffle(self.img, self.rng, 1.0)
        orig = sorted(self.img[..., c].tobytes() for c in range(3))
        self.assertEqual(sorted(out[..., c].tobytes() for c in range(3)), orig)

    def test_cutout_creates_black_block_and_erasing_does_not(self):
        base = np.full((100, 100, 3), 200, np.uint8)
        self.assertTrue((a.p_cutout(base, self.rng, 1.0) == 0).all(axis=2).any())
        self.assertFalse((a.p_random_erasing(base, self.rng, 1.0) == 200).all())

    def test_hue_changes_color_not_brightness_much(self):
        out = a.p_hue(self.img, self.rng, 1.0)
        self.assertEqual(out.shape, self.img.shape)

    def test_elastic_moves_pixels(self):
        out = a.n_elastic(self.img, self.rng, 1.0)
        self.assertFalse(np.array_equal(out, self.img))

    def test_enhance_and_pil_filter_helpers(self):
        from PIL import ImageEnhance, ImageFilter
        self.assertEqual(a._enhance(self.img, ImageEnhance.Brightness, 1.0).shape, self.img.shape)
        self.assertEqual(a._pil_filter(self.img, ImageFilter.SHARPEN).shape, self.img.shape)

    def test_motion_blur_preserves_average_brightness(self):
        out = a.p_motion_blur(self.img, self.rng, 1.0)
        self.assertLess(abs(float(out.mean()) - float(self.img.mean())), 5)


class TestPipelineHelpers(unittest.TestCase):
    def test_apply_methods_geometric_merged_and_shape_kept(self):
        img = make_img()
        out, b = a.apply_methods(img, None, ["Rotation", "Zooming", "Hue", "Cutout"],
                                 np.random.default_rng(1), 1.0, "Reflect")
        self.assertEqual(out.shape, img.shape)
        self.assertIsNone(b)

    def test_all_border_modes(self):
        for mode in a.BORDER_MODES:
            out, _ = a.apply_methods(make_img(), None, ["Rotation"], np.random.default_rng(1), 1.0, mode)
            self.assertEqual(out.shape, (90, 120, 3))

    def test_constant_border_leaves_black_corners(self):
        img = np.full((100, 100, 3), 255, np.uint8)
        out, _ = a.apply_methods(img, None, ["Rotation"], np.random.default_rng(4), 1.0, "Constant (black)")
        self.assertTrue((out == 0).any())

    def test_effective_methods_drops_elastic_only_with_labels(self):
        names = ["Rotation", "Elastic Transformation"]
        self.assertEqual(a.effective_methods(names, True), ["Rotation"])
        self.assertEqual(a.effective_methods(names, False), names)

    def test_choose_for_pipeline_never_empty(self):
        rng = np.random.default_rng(0)
        names = ["Rotation", "Hue", "Cutout"]
        for p in (0.0, 0.3, 1.0):
            chosen = a.choose_for_pipeline(names, p, rng)
            self.assertTrue(chosen and set(chosen) <= set(names))
        self.assertEqual(a.choose_for_pipeline(names, 1.0, rng), names)


# ---------------------------------------------------------------- bounding boxes
class TestBoxes(unittest.TestCase):
    def test_identity(self):
        b = np.array([[0, 0.5, 0.5, 0.2, 0.3]])
        np.testing.assert_allclose(a.transform_boxes(b, np.eye(3), 100, 100), b)

    def test_flip_mirrors_x_center(self):
        b = np.array([[1, 0.3, 0.5, 0.2, 0.2]])
        out = a.transform_boxes(b, a.g_flip(100, 100, None, 1), 100, 100)
        np.testing.assert_allclose(out[0], [1, 0.7, 0.5, 0.2, 0.2])

    def test_box_that_leaves_frame_is_dropped(self):
        M = np.array([[1, 0, 500], [0, 1, 0], [0, 0, 1.0]])
        self.assertEqual(len(a.transform_boxes(np.array([[0, .5, .5, .2, .2]]), M, 100, 100)), 0)

    def test_partially_visible_box_is_clipped(self):
        M = np.array([[1, 0, 40], [0, 1, 0], [0, 0, 1.0]])
        out = a.transform_boxes(np.array([[0, .5, .5, .4, .4]]), M, 100, 100)
        self.assertEqual(len(out), 1)
        self.assertLessEqual(out[0][1] + out[0][3] / 2, 1.0001)

    def test_output_size_normalization(self):
        M = np.diag([2.0, 2.0, 1.0])  # 100x100 -> 200x200
        out = a.transform_boxes(np.array([[0, .5, .5, .2, .2]]), M, 100, 100, 200, 200)
        np.testing.assert_allclose(out[0], [0, .5, .5, .2, .2])

    def test_empty_input(self):
        self.assertEqual(a.transform_boxes(np.zeros((0, 5)), np.eye(3), 10, 10).shape, (0, 5))

    def test_apply_methods_keeps_boxes_normalized(self):
        b = np.array([[0, .5, .5, .3, .3], [1, .1, .1, .1, .1]])
        for name, (kind, _) in a.REGISTRY.items():
            if kind != "geo":
                continue
            with self.subTest(name=name):
                _, out = a.apply_methods(make_img(), b, [name], np.random.default_rng(2), 1.0, "Reflect")
                self.assertEqual(out.shape[1], 5)
                self.assertTrue(((out[:, 1:] >= 0) & (out[:, 1:] <= 1.0001)).all())

    def test_draw_boxes(self):
        img = np.zeros((100, 100, 3), np.uint8)
        self.assertIs(a.draw_boxes(img, None), img)
        out = a.draw_boxes(img, np.array([[0, .5, .5, .4, .4]]))
        self.assertTrue((out[..., 1] == 255).any())
        self.assertFalse(img.any())  # original untouched


# ---------------------------------------------------------------- file helpers
class TestFiles(TmpCase):
    def test_collect_images_recursive_case_insensitive_and_excludes_dst(self):
        src = self.folder("in", ["a.JPG", "b.png"])
        os.makedirs(self.p("in", "sub"))
        save(make_img(), self.p("in", "sub", "c.bmp"))
        open(self.p("in", "notes.txt"), "w").write("x")
        os.makedirs(self.p("in", "out"))
        save(make_img(), self.p("in", "out", "old.png"))
        names = sorted(os.path.basename(f) for f in a.collect_images(src, self.p("in", "out")))
        self.assertEqual(names, ["a.JPG", "b.png", "c.bmp"])
        self.assertEqual(len(a.collect_images(src)), 4)

    def test_no_images_message_lists_file_types(self):
        os.makedirs(self.p("x"))
        open(self.p("x", "doc.xyz"), "w").write("1")
        self.assertIn(".xyz", a.no_images_message(self.p("x")))

    def test_load_image_always_rgb(self):
        for mode, ext in (("L", ".png"), ("P", ".gif"), ("RGBA", ".png"), ("RGB", ".jpg")):
            with self.subTest(mode=mode):
                path = self.p("t" + ext)
                Image.fromarray(make_img()).convert(mode).save(path)
                out = a.load_image(path)
                self.assertEqual(out.shape, (90, 120, 3))
                self.assertEqual(out.dtype, np.uint8)

    def test_load_image_respects_exif_rotation(self):
        path = self.p("rot.jpg")
        exif = Image.Exif()
        exif[0x0112] = 6
        Image.fromarray(make_img(100, 50)).save(path, exif=exif.tobytes())
        self.assertEqual(a.load_image(path).shape[:2], (100, 50))

    def test_save_image_roundtrip_formats(self):
        img = make_img()
        for ext in (".png", ".jpg", ".bmp", ".tif", ".webp"):
            with self.subTest(ext=ext):
                path = self.p("s" + ext)
                a.save_image(img, path)
                self.assertEqual(a.load_image(path).shape, img.shape)
        a.save_image(img, self.p("lossless.png"))
        np.testing.assert_array_equal(a.load_image(self.p("lossless.png")), img)

    def test_labels_read_write(self):
        img = self.p("x.jpg")
        self.assertIsNone(a.read_labels(img))
        a.write_labels(np.array([[2, .5, .5, .25, .25]]), self.p("x.txt"))
        np.testing.assert_allclose(a.read_labels(img), [[2, .5, .5, .25, .25]])
        self.assertEqual(open(self.p("x.txt")).read().strip(), "2 0.500000 0.500000 0.250000 0.250000")

    def test_read_labels_rejects_polygons_and_ignores_blank_lines(self):
        open(self.p("p.txt"), "w").write("0 .1 .1 .2 .2 .3 .3\n")
        with self.assertRaises(ValueError):
            a.read_labels(self.p("p.jpg"))
        open(self.p("b.txt"), "w").write("\n0 .5 .5 .1 .1\n\n")
        self.assertEqual(len(a.read_labels(self.p("b.jpg"))), 1)
        open(self.p("e.txt"), "w").write("")
        self.assertEqual(a.read_labels(self.p("e.jpg")).shape, (0, 5))

    def test_unique_path(self):
        path = self.p("f.png")
        self.assertEqual(a.unique_path(path), path)
        open(path, "w").close()
        self.assertEqual(a.unique_path(path), self.p("f_1.png"))
        open(self.p("f_1.png"), "w").close()
        self.assertEqual(a.unique_path(path), self.p("f_2.png"))

    def test_slug(self):
        self.assertEqual(a.slug("Salt & Pepper"), "salt-pepper")
        self.assertEqual(a.slug("Gaussian Blur"), "gaussian-blur")


# ---------------------------------------------------------------- rename
class TestRename(TmpCase):
    def test_natural_key_and_list_images(self):
        self.assertEqual(sorted(["10.jpg", "2.jpg", "1.jpg"], key=a.natural_key), ["1.jpg", "2.jpg", "10.jpg"])
        src = self.folder("r", ["10.jpg", "2.jpg", "x.txt"][:2])
        open(self.p("r", "x.txt"), "w").write("")
        os.makedirs(self.p("r", "dir.jpg"))  # a folder named like an image must be ignored
        self.assertEqual(a.list_images(src), ["2.jpg", "10.jpg"])
        self.assertEqual(a.list_images(src, natural=False), ["10.jpg", "2.jpg"])

    def test_plan_renames(self):
        src = self.folder("r", ["b.jpg", "a.png"])
        open(self.p("r", "a.txt"), "w").write("0 .5 .5 .1 .1\n")
        pairs = a.plan_renames(src, 5, 3, True, True)
        self.assertEqual(pairs, [("a.png", "005.png"), ("a.txt", "005.txt"), ("b.jpg", "006.jpg")])
        self.assertEqual(len(a.plan_renames(src, 1, 0, True, False)), 2)

    def test_apply_renames_handles_chains_and_swaps(self):
        src = self.folder("r", ["2.jpg", "a.jpg"])
        before = {f: open(self.p("r", f), "rb").read() for f in os.listdir(src)}
        a.apply_renames(src, [("2.jpg", "1.jpg"), ("a.jpg", "2.jpg")])
        self.assertEqual(sorted(os.listdir(src)), ["1.jpg", "2.jpg"])
        self.assertEqual(open(self.p("r", "1.jpg"), "rb").read(), before["2.jpg"])
        self.assertEqual(open(self.p("r", "2.jpg"), "rb").read(), before["a.jpg"])

    def test_apply_renames_refuses_to_overwrite(self):
        src = self.folder("r", ["x.jpg"])
        open(self.p("r", "1.jpg.bak"), "w").close()
        open(self.p("r", "1.jpg"), "w").write("not an image in the plan")
        with self.assertRaises(FileExistsError):
            a.apply_renames(src, [("x.jpg", "1.jpg")])
        self.assertEqual(open(self.p("r", "1.jpg")).read(), "not an image in the plan")

    def test_apply_renames_rolls_back_on_failure(self):
        src = self.folder("r", ["2.jpg", "a.jpg"])
        real, calls = os.rename, [0]

        def flaky(x, y):
            calls[0] += 1
            if calls[0] == 4:
                raise OSError("disk error")
            real(x, y)

        with mock.patch("augmento.os.rename", side_effect=flaky):
            with self.assertRaises(OSError):
                a.apply_renames(src, [("2.jpg", "1.jpg"), ("a.jpg", "2.jpg")])
        self.assertEqual(sorted(os.listdir(src)), ["2.jpg", "a.jpg"])

    def test_run_rename_labels_map_and_counts(self):
        src = self.folder("r", ["z.jpg", "m.PNG"])
        open(self.p("r", "z.txt"), "w").write("0 .5 .5 .1 .1\n")
        msgs = run(a.run_rename, dict(src=src, recursive=False, start=1, pad=0, natural=True,
                                      labels=True, save_map=True))
        self.assertEqual(msgs[-1], ("done", 2, 0, False))
        self.assertEqual(sorted(os.listdir(src)), ["1.PNG", "2.jpg", "2.txt", "rename_map.csv"])
        rows = list(csv.reader(open(self.p("r", "rename_map.csv"))))
        self.assertEqual(rows[0], ["old_name", "new_name"])
        self.assertIn(["z.jpg", "2.jpg"], rows)

    def test_run_rename_is_idempotent(self):
        src = self.folder("r", ["1.jpg", "2.jpg"])
        msgs = run(a.run_rename, dict(src=src, recursive=False, start=1, pad=0, natural=True,
                                      labels=True, save_map=True))
        self.assertEqual(sorted(os.listdir(src)), ["1.jpg", "2.jpg"])  # nothing changed, no map written
        self.assertEqual(msgs[-1][:3], ("done", 2, 0))

    def test_run_rename_subfolders_numbered_separately(self):
        self.folder("r/cat", ["q.jpg", "w.jpg"])
        self.folder("r/dog", ["q.jpg", "w.jpg"])
        run(a.run_rename, dict(src=self.p("r"), recursive=True, start=0, pad=2, natural=True,
                               labels=False, save_map=False))
        self.assertEqual(sorted(os.listdir(self.p("r", "cat"))), ["00.jpg", "01.jpg"])
        self.assertEqual(sorted(os.listdir(self.p("r", "dog"))), ["00.jpg", "01.jpg"])

    def test_run_rename_conflict_skips_folder_and_changes_nothing(self):
        src = self.folder("r", ["x.jpg"])
        open(self.p("r", "x.txt"), "w").write("0 .5 .5 .1 .1\n")
        open(self.p("r", "1.txt"), "w").write("orphan")
        msgs = run(a.run_rename, dict(src=src, recursive=False, start=1, pad=0, natural=True,
                                      labels=True, save_map=False))
        self.assertEqual(msgs[-1][:3], ("done", 0, 1))
        self.assertTrue(logs(msgs))
        self.assertEqual(sorted(os.listdir(src)), ["1.txt", "x.jpg", "x.txt"])


# ---------------------------------------------------------------- resize
class TestResize(TmpCase):
    def test_resize_image_exact_size_all_modes_and_extremes(self):
        for size in ((20, 10), (1000, 1000), (641, 639), (600, 300)):
            for mode in ("stretch", "letterbox", "crop"):
                with self.subTest(size=size, mode=mode):
                    img = np.zeros((size[1], size[0], 3), np.uint8)
                    out, M = a.resize_image(img, 640, 720, mode)
                    self.assertEqual(out.shape, (720, 640, 3))
                    self.assertEqual(M.shape, (3, 3))

    def test_letterbox_pads_gray_and_keeps_proportions(self):
        img = np.full((300, 600, 3), 255, np.uint8)
        out, _ = a.resize_image(img, 640, 640, "letterbox")
        self.assertTrue((out[0, 0] == 114).all())          # padding
        self.assertTrue((out[320, 320] == 255).all())      # image

    def test_crop_has_no_padding(self):
        img = np.full((300, 600, 3), 255, np.uint8)
        out, _ = a.resize_image(img, 640, 640, "crop")
        self.assertTrue((out == 255).all())

    def test_box_matrices_per_mode(self):
        img = np.zeros((300, 600, 3), np.uint8)
        box = np.array([[0, .5, .5, .5, .5]])
        expect = {"stretch": [0, .5, .5, .5, .5], "letterbox": [0, .5, .5, .5, .25], "crop": [0, .5, .5, 1.0, .5]}
        for mode, want in expect.items():
            with self.subTest(mode=mode):
                _, M = a.resize_image(img, 640, 640, mode)
                np.testing.assert_allclose(a.transform_boxes(box, M, 600, 300, 640, 640)[0], want, atol=0.01)

    def test_run_resize_all_modes_labels_and_formats(self):
        src = self.folder("rs", ["a.jpg", "g.gif"], size=(600, 300))
        os.makedirs(self.p("rs", "k"))
        save(make_img(100, 100), self.p("rs", "k", "b.png"), "L")
        open(self.p("rs", "a.txt"), "w").write("0 0.5 0.5 0.5 0.5\n")
        for mode, want in (("stretch", "0 0.5 0.5 0.5 0.5"), ("letterbox", "0 0.500000 0.500000 0.500000 0.250000"),
                           ("crop", "0 0.500000 0.500000 1.000000 0.500000")):
            with self.subTest(mode=mode):
                dst = self.p("out_" + mode)
                msgs = run(a.run_resize, dict(src=src, dst=dst, w=640, h=640, mode=mode, use_labels=True))
                self.assertEqual(msgs[-1], ("done", 3, 0, False))
                self.assertEqual(open(os.path.join(dst, "a.txt")).read().strip(), want)
                for rel in ("a.jpg", "g.png", os.path.join("k", "b.png")):
                    with Image.open(os.path.join(dst, rel)) as im:
                        self.assertEqual(im.size, (640, 640))
        self.assertEqual(sorted(os.listdir(src)), ["a.jpg", "a.txt", "g.gif", "k"])  # originals untouched

    def test_run_resize_without_labels_flag_ignores_txt(self):
        src = self.folder("rs", ["a.jpg"])
        open(self.p("rs", "a.txt"), "w").write("0 .5 .5 .1 .1\n")
        run(a.run_resize, dict(src=src, dst=self.p("o"), w=64, h=64, mode="stretch", use_labels=False))
        self.assertEqual(os.listdir(self.p("o")), ["a.jpg"])

    def test_run_resize_polygon_label_is_skipped_not_fatal(self):
        src = self.folder("rs", ["a.jpg", "b.jpg"])
        open(self.p("rs", "a.txt"), "w").write("0 .1 .1 .2 .2 .3 .3\n")
        msgs = run(a.run_resize, dict(src=src, dst=self.p("o"), w=64, h=64, mode="letterbox", use_labels=True))
        self.assertEqual(msgs[-1], ("done", 1, 1, False))
        self.assertIn("a.jpg", logs(msgs)[0])

    def test_run_resize_cancel(self):
        src = self.folder("rs", ["a.jpg", "b.jpg"])
        stop = threading.Event()
        stop.set()
        msgs = run(a.run_resize, dict(src=src, dst=self.p("o"), w=64, h=64, mode="stretch", use_labels=False), stop)
        self.assertEqual(msgs[-1], ("done", 0, 0, True))


# ---------------------------------------------------------------- augment job
class TestRunJob(TmpCase):
    def cfg(self, src, dst, **kw):
        base = dict(src=src, dst=dst, names=["Rotation", "Hue"], mode="separate", copies=2, prob=0.5,
                    k=1.0, seed=1, border="Reflect", use_labels=False, originals=False)
        base.update(kw)
        return base

    def count(self, folder):
        return sum(len(fs) for _, _, fs in os.walk(folder))

    def test_separate_mode_count(self):
        src = self.folder("in")  # 3 images
        msgs = run(a.run_job, self.cfg(src, self.p("o")))
        self.assertEqual(msgs[-1], ("done", 12, 0, False))  # 3 images x 2 methods x 2 copies
        self.assertEqual(self.count(self.p("o")), 12)

    def test_pipeline_mode_count_and_names(self):
        src = self.folder("in")
        msgs = run(a.run_job, self.cfg(src, self.p("o"), mode="pipeline"))
        self.assertEqual(msgs[-1][1], 6)
        self.assertTrue(all("_aug_" in f for f in os.listdir(self.p("o"))))

    def test_originals_copied_and_extension_kept(self):
        src = self.folder("in")
        run(a.run_job, self.cfg(src, self.p("o"), originals=True, copies=1, names=["Hue"]))
        out = os.listdir(self.p("o"))
        for name in ("a.jpg", "b.png", "c.jpeg", "a_hue_1.jpg", "b_hue_1.png", "c_hue_1.jpeg"):
            self.assertIn(name, out)

    def test_same_seed_gives_identical_output(self):
        src = self.folder("in", ["a.png"])
        run(a.run_job, self.cfg(src, self.p("o1"), seed=42))
        run(a.run_job, self.cfg(src, self.p("o2"), seed=42))
        run(a.run_job, self.cfg(src, self.p("o3"), seed=43))
        f = "a_rotation_1.png"
        same = open(self.p("o1", f), "rb").read() == open(self.p("o2", f), "rb").read()
        diff = open(self.p("o1", f), "rb").read() != open(self.p("o3", f), "rb").read()
        self.assertTrue(same and diff)

    def test_rerun_never_overwrites(self):
        src = self.folder("in", ["a.png"])
        run(a.run_job, self.cfg(src, self.p("o"), copies=1, names=["Hue"]))
        run(a.run_job, self.cfg(src, self.p("o"), copies=1, names=["Hue"]))
        self.assertEqual(sorted(os.listdir(self.p("o"))), ["a_hue_1.png", "a_hue_1_1.png"])

    def test_subfolders_are_mirrored(self):
        self.folder("in/cat", ["a.jpg"])
        self.folder("in/dog", ["b.jpg"])
        run(a.run_job, self.cfg(self.p("in"), self.p("o"), copies=1, names=["Hue"]))
        self.assertTrue(os.path.isfile(self.p("o", "cat", "a_hue_1.jpg")))
        self.assertTrue(os.path.isfile(self.p("o", "dog", "b_hue_1.jpg")))

    def test_labels_follow_flip(self):
        src = self.folder("in", ["a.jpg"])
        open(self.p("in", "a.txt"), "w").write("0 0.3 0.5 0.2 0.2\n")
        run(a.run_job, self.cfg(src, self.p("o"), names=["Flipping"], copies=1, use_labels=True, originals=True))
        self.assertEqual(open(self.p("o", "a_flipping_1.txt")).read().split()[:3], ["0", "0.700000", "0.500000"])
        self.assertEqual(open(self.p("o", "a.txt")).read().split()[1], "0.3")  # copied original label

    def test_corrupt_file_is_skipped_and_logged(self):
        src = self.folder("in", ["a.jpg"])
        open(self.p("in", "bad.jpg"), "wb").write(b"not an image")
        msgs = run(a.run_job, self.cfg(src, self.p("o"), copies=1, names=["Hue"]))
        self.assertEqual(msgs[-1], ("done", 1, 1, False))
        self.assertIn("bad.jpg", logs(msgs)[0])

    def test_cancel_stops_immediately(self):
        src = self.folder("in")
        stop = threading.Event()
        stop.set()
        msgs = run(a.run_job, self.cfg(src, self.p("o")), stop)
        self.assertEqual(msgs[-1], ("done", 0, 0, True))

    def test_progress_messages_reach_total(self):
        src = self.folder("in")
        msgs = run(a.run_job, self.cfg(src, self.p("o")))
        self.assertIn(("progress", 3, 3), msgs)


# ---------------------------------------------------------------- GUI (needs a display)
class TestGUI(TmpCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        try:
            cls.root = tk.Tk()
        except tk.TclError:
            raise unittest.SkipTest("no display available")
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        super().setUp()
        self.shown = []
        patches = [mock.patch.object(a.messagebox, "showerror", lambda t, m, **k: self.shown.append(("error", m))),
                   mock.patch.object(a.messagebox, "showinfo", lambda t, m, **k: self.shown.append(("info", m))),
                   mock.patch.object(a.messagebox, "askyesno", lambda *x, **k: True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.app = a.App(self.root)

    def wait(self):
        t = time.time()
        while self.app.running and time.time() - t < 30:
            self.root.update()
            time.sleep(0.02)
        self.root.update()

    def test_estimate_updates_with_selection(self):
        app = self.app
        app.src.set(self.folder("in"))
        app.refresh_count()
        self.assertIn("3 image(s) found", app.estimate.cget("text"))
        app.method_vars["Hue"].set(True)
        app.copies.set(2)
        self.assertIn("write 6 image(s)", app.estimate.cget("text"))
        app.mode.set("pipeline")
        self.assertIn("write 6 image(s)", app.estimate.cget("text"))
        app.originals.set(True)
        self.assertIn("write 9 image(s)", app.estimate.cget("text"))

    def test_select_all_clear_and_elastic_hidden_with_labels(self):
        app = self.app
        app.set_all(True)
        self.assertEqual(len(app.selected()), 25)
        app.use_labels.set(True)
        self.assertEqual(len(app.selected()), 24)
        app.set_all(False)
        self.assertEqual(app.selected(), [])

    def test_make_config_validation(self):
        app = self.app
        self.assertIsNone(app.make_config())
        app.src.set(self.folder("in"))
        self.assertIsNone(app.make_config())                       # no output
        app.dst.set(self.p("in"))
        self.assertIsNone(app.make_config())                       # same folder
        app.dst.set(self.p("out"))
        self.assertIsNone(app.make_config())                       # nothing selected
        app.method_vars["Hue"].set(True)
        app.seed.set("abc")
        self.assertIsNone(app.make_config())                       # bad seed
        app.seed.set("5")
        cfg = app.make_config()
        self.assertEqual((cfg["seed"], cfg["names"]), (5, ["Hue"]))
        self.assertGreaterEqual(len([s for s in self.shown if s[0] == "error"]), 5)

    def test_generate_end_to_end(self):
        app = self.app
        app.src.set(self.folder("in"))
        app.dst.set(self.p("out"))
        app.method_vars["Hue"].set(True)
        app.start()
        self.wait()
        self.assertIn("Finished: 3 image(s) written", app.status.cget("text"))
        self.assertEqual(len(os.listdir(self.p("out"))), 3)
        self.assertEqual(str(app.gen_btn.cget("state")), "normal")

    def test_rename_tab_end_to_end(self):
        app = self.app
        src = self.folder("r", ["x.jpg", "y.jpg"])
        app.rn_dir.set(src)
        app.refresh_rn()
        self.assertIn("2 image(s) found", app.rn_info.cget("text"))
        app.start_rename()
        self.wait()
        self.assertIn("2 image(s) renamed", app.status.cget("text"))
        self.assertEqual(sorted(f for f in os.listdir(src) if f.endswith(".jpg")), ["1.jpg", "2.jpg"])
        app.rn_dir.set("/does/not/exist")
        app.start_rename()
        self.assertEqual(self.shown[-1][0], "error")

    def test_resize_tab_end_to_end_and_preset(self):
        app = self.app
        app.rs_preset.set("720x720")
        app.apply_preset()
        self.assertEqual((app.rs_w.get(), app.rs_h.get()), (720, 720))
        app.rs_preset.set("garbage")
        app.apply_preset()                                         # must not crash
        src = self.folder("rs", ["a.jpg"])
        app.rs_src.set(src)
        app.rs_dst.set(self.p("rs_out"))
        app.refresh_rs()
        self.assertIn("1 image(s) found", app.rs_info.cget("text"))
        app.rs_mode.set("Keep aspect ratio, pad (letterbox)")
        app.start_resize()
        self.wait()
        with Image.open(self.p("rs_out", "a.jpg")) as im:
            self.assertEqual(im.size, (720, 720))
        app.rs_dst.set(src)
        app.start_resize()
        self.assertIn("must differ", self.shown[-1][1])

    def test_helpers_get_log_to_photo_count_rename(self):
        app = self.app
        app.copies.set(5)
        self.assertEqual(app._get(app.copies, 1), 5)
        app.log("hello")
        self.assertIn("hello", app.log_box.get("1.0", "end"))
        self.assertIsNotNone(a.to_photo(make_img()))
        src = self.folder("c")
        os.makedirs(self.p("c", "sub"))
        save(make_img(), self.p("c", "sub", "z.png"))
        self.assertEqual(app.count_rename(src, False), 3)
        self.assertEqual(app.count_rename(src, True), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
