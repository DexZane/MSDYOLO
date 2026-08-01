"""Regression checks for the real training entry point."""

from msdyolo.train import validatepaths


class CheckTrainEntry:

    def checkvalidatepathsacceptsvalidfiles(self, tmp_path):
        paths = {}
        for key in ("cfg", "data", "hyp"):
            path = tmp_path / f"{key}.yaml"
            path.write_text("{}\n", encoding="utf-8")
            paths[key] = str(path)
        paths["weights"] = ""

        validatepaths(paths)
