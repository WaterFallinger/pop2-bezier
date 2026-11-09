import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
import render_csv as replay

HEADER = ["image_name", "cx", "cy", "w", "h", "shape_type", "theta", "r", "g", "b", "is_region_marker", "stage", "region_id"]
class ReplayTests(unittest.TestCase):
    def write(self, root, rows):
        with (root / "data_part_1.csv").open("w", newline="", encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(HEADER);w.writerows(rows)
    def test_background_markers_and_blending(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.write(root, [["sample",0,0,0,0,-1,0,100,20,0,0,"global",-1], ["sample",64,64,128,128,2,0,0,0,0,1,"coarse",0], ["sample",64,64,20,20,0,0,200,0,0,0,"coarse",0], ["sample",64,64,128,128,4,0,0,0,0,1,"coarse",0]])
            group=replay.read_groups(root)["sample"];image=replay.render(group)
            self.assertEqual(group["marker_count"],2)
            self.assertTrue(np.array_equal(image[64,64], [150,10,0]))
            self.assertTrue(np.array_equal(image[0,0], [100,20,0]))
            self.assertEqual(image.shape,(512,512,3))
    def test_missing_background_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.write(root, [["sample",64,64,20,20,0,0,200,0,0,0,"global",-1]])
            with self.assertRaisesRegex(ValueError,"background"):replay.read_groups(root)
if __name__=="__main__":unittest.main()
