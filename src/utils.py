"""
Utility functions for date conversion and year-fraction calculations.

Provides Python equivalents for MATLAB's datenum system and
day-count conventions used in fixed-income pricing.

Approach notes:
    - MATLAB datenum = Python date.toordinal() + 366
    - yearfrac(d1, d2, 6) [30/360 European] is reimplemented as yearfrac_30_360
    - yearfrac(d1, d2, 3) [ACT/365 Fixed] is reimplemented as yearfrac_act_365
"""

import numpy as np
import pandas as pd
from datetime import date, datetime


def to_datenum(d):
    """Convert a Python date or datetime to a MATLAB serial date number."""
    if isinstance(d, datetime):
        d = d.date()
    if isinstance(d, pd.Timestamp):
        d = d.date()
    return d.toordinal() + 366


def from_datenum(dn):
    """Convert a MATLAB serial date number to a Python date."""
    return date.fromordinal(int(dn) - 366)


def to_datenum_array(dates):
    """Convert an array-like of dates to an array of MATLAB datenums."""
    out = np.zeros(len(dates), dtype=np.float64)
    for i, d in enumerate(dates):
        out[i] = to_datenum(d)
    return out


def from_datenum_array(dns):
    """Convert an array of MATLAB datenums to a list of Python dates."""
    return [from_datenum(int(dn)) for dn in dns]


def yearfrac_30_360(d1, d2):
    """
    Year fraction using the 30/360 European convention.

    Matches MATLAB yearfrac(d1, d2, 6).  Both d1 and d2 may be
    scalars (single dates) or equal-length iterables.

    Returns:
        float or np.ndarray
    """
    if isinstance(d1, (date, datetime, pd.Timestamp)):
        d1 = [d1]
        d2 = [d2]
        scalar = True
    else:
        scalar = False

    result = np.zeros(len(d1))
    for i in range(len(d1)):
        dd1, dd2 = pd.Timestamp(d1[i]), pd.Timestamp(d2[i])
        y1, m1, day1 = dd1.year, dd1.month, min(dd1.day, 30)
        y2, m2, day2 = dd2.year, dd2.month, dd2.day

        if day2 == 31 and day1 >= 30:
            day2 = 30
        if day1 == 31:
            day1 = 30

        result[i] = (360 * (y2 - y1) + 30 * (m2 - m1) + (day2 - day1)) / 360.0

    return result[0] if scalar else result


def yearfrac_act_365(d1, d2):
    """
    Year fraction using the ACT/365 Fixed convention.

    Matches MATLAB yearfrac(d1, d2, 3).  Both d1 and d2 may be
    scalars or equal-length iterables.

    Returns:
        float or np.ndarray
    """
    if isinstance(d1, (date, datetime, pd.Timestamp)):
        d1_arr = [pd.Timestamp(d1)]
        d2_arr = [pd.Timestamp(d2)]
        scalar = True
    else:
        d1_arr = [pd.Timestamp(x) for x in d1]
        d2_arr = [pd.Timestamp(x) for x in d2]
        scalar = False

    result = np.zeros(len(d1_arr))
    for i in range(len(d1_arr)):
        delta = (d2_arr[i] - d1_arr[i]).days
        result[i] = delta / 365.0

    return result[0] if scalar else result
