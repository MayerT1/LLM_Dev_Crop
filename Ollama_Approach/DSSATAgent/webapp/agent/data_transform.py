"""
This module contains funtions to transform data. Transformation implies converting
between data formats, or extrating data from some file to create a new file.
"""
import pandas as pd

import re


ENV_PARSE_INDEX = (
    "Emergence-End Juvenile", "End Juvenil-Floral Init",
    "Floral Init-End Lf Grow", "End Lf Grth-Beg Grn Fil",
    "Grain Filling Phase", "Planting to Harvest"
)
ENV_STRESS_COLNAMES = (
    "devPhase", "timeSpan", "avgTmax", "avgTmin", "avgTmean", "avgSrad", "avgPhotper", "avgCO2",
    "cummRain", "cummET", "cummETp", "ndaysTminLt0", "ndaysTminLt2", "ndaysTmaxGt30",
    "ndaysTmaxGt32", "ndaysTmaxGt34", "ndaysRainGt0", "stressWatPho", "stressWatGro",
    "stressNitPhto", "stressNitGro", "stressPhoPho", "stressPhoGro"
)


def parse_overview(overview_str):
    """
    Parse the overview file to get environmental and stress factors
    """
    lines = []
    for key in ENV_PARSE_INDEX:
        for n, l in enumerate(re.findall(f"({key}.+)\n", overview_str), 1):
            lines.append(
                [n, key] + l.replace(key, "").split()
            )
    df = pd.DataFrame(lines, columns=["RUN"] + list(ENV_STRESS_COLNAMES))
    return df