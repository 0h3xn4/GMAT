#!/usr/bin/env python3
"""Orbit lifetime after the disposal of the FEEP formation satellites.

Checks the end-of-life requirement of ESSB-ST-U-007 5.4.2.3.a (natural orbit
lifetime shorter than 5 years from the end of life) the way its Clause 6.2
asks for it: Cd 2.2 and the 50th percentile of the solar activity (6.2.g, AD10
5.9), with the start epoch moved through a full solar cycle (6.2.f). It runs one
GMAT free-decay propagation per (altitude, start year) with GmatConsole and
prints a table of lifetimes. The first altitude/year of the table is the
disposal orbit and end-of-life epoch of Ex_FormationFlying_SSO_FEEP.script
(HDisp, end of life 31 Dec 2031, decay starting after the thrusting descent); use it
to choose HDisp. At 1 mN the descent lasts about 270 days: use --epoch-date "27 Sep"
--descent-days 270 and --first-year 2032.

    python formation_eol_lifetime.py --gmat-bin /path/to/GMAT/bin/GmatConsole \\
        --altitudes 440 450 460 --first-year 2032 --years 11 \\
        --epoch-date "27 Sep" --descent-days 270 --jobs 4

The mean altitude is that of the disposal orbit (circular, e = 0.0012). Area, mass
and inclination default to the values of the script at the end of life.
Needs only the Python standard library.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

TEMPLATE = """
Create Spacecraft Sat;
Sat.DateFormat       = UTCGregorian;
Sat.Epoch            = '{epoch}';
Sat.CoordinateSystem = EarthTOD;
Sat.DisplayStateType = BrouwerMeanShort;
Sat.BrouwerShortSMA  = {sma};
Sat.BrouwerShortECC  = 0.0012;
Sat.BrouwerShortINC  = {inc};
Sat.BrouwerShortRAAN = 258.7256;
Sat.BrouwerShortAOP  = 90;
Sat.BrouwerShortMA   = 0;
Sat.DryMass          = {mass};
Sat.Cd               = {cd};
Sat.Cr               = 1.3;
Sat.DragArea         = {area};
Sat.SRPArea          = {area};
Create ForceModel FM;
FM.CentralBody                      = Earth;
FM.PrimaryBodies                    = {{Earth}};
FM.PointMasses                      = {{Sun, Luna}};
FM.SRP                              = On;
FM.ErrorControl                     = RSSStep;
FM.GravityField.Earth.Degree        = 4;
FM.GravityField.Earth.Order         = 4;
FM.GravityField.Earth.PotentialFile = 'EGM96.cof';
FM.GravityField.Earth.TideModel     = 'None';
FM.SRP.Flux                         = 1367;
FM.SRP.SRPModel                     = Spherical;
FM.SRP.Nominal_Sun                  = 149597870.691;
FM.Drag.AtmosphereModel             = NRLMSISE00;
FM.Drag.HistoricWeatherSource       = 'ConstantFluxAndGeoMag';
FM.Drag.PredictedWeatherSource      = 'SchattenFile';
FM.Drag.SchattenFile                = 'SchattenPredict.txt';
FM.Drag.SchattenErrorModel          = '{pct}';
FM.Drag.SchattenTimingModel         = 'NominalCycle';
FM.Drag.F107                        = 150;
FM.Drag.F107A                       = 150;
FM.Drag.MagneticIndex               = 3;
Create Propagator Prop;
Prop.FM                       = FM;
Prop.Type                     = PrinceDormand78;
Prop.InitialStepSize          = 60;
Prop.Accuracy                 = 1e-9;
Prop.MinStep                  = 0.001;
Prop.MaxStep                  = 600;
Prop.MaxStepAttempts          = 50;
Prop.StopIfAccuracyIsViolated = false;
Create CoordinateSystem EarthTOD;
EarthTOD.Origin = Earth;
EarthTOD.Axes   = TODEq;
Create Variable lifeDays t0;
Create ReportFile R;
R.Filename     = '{out}';
R.WriteHeaders = false;
R.ColumnWidth  = 18;
R.FixedWidth   = true;
R.LeftJustify  = On;
R.Precision    = 8;
BeginMissionSequence;
t0 = Sat.A1ModJulian;
Propagate Prop(Sat) {{Sat.ElapsedDays = {maxdays}, Sat.Earth.Altitude = 120}};
lifeDays = Sat.A1ModJulian - t0;
Report R lifeDays;
"""


def run_case(args, alt, year):
    tag = f"life_{alt:g}_{year}"
    out = tag + ".txt"
    script = TEMPLATE.format(
        epoch=f"{args.epoch_date} {year} 00:00:00.000", sma=6378.1363 + alt, inc=args.inc,
        mass=args.mass, cd=args.cd, area=args.area, pct=args.percentile,
        out=out, maxdays=args.max_days)
    gmat = os.path.abspath(args.gmat_bin)
    outdir = os.path.join(os.path.dirname(gmat), "..", "output")
    path = os.path.join(outdir, out)
    if os.path.exists(path):
        os.remove(path)
    with tempfile.NamedTemporaryFile("w", suffix=".script", delete=False) as f:
        f.write(script)
        name = f.name
    try:
        subprocess.run([gmat, "--run", name], cwd=os.path.dirname(gmat),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        os.remove(name)
    try:
        return float(open(path).read().split()[0])
    except (OSError, IndexError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmat-bin", required=True, help="path of GmatConsole")
    ap.add_argument("--altitudes", type=float, nargs="+", default=[480, 500, 520],
                    help="mean altitudes of the disposal orbit, km")
    ap.add_argument("--first-year", type=int, default=2031,
                    help="first end-of-life epoch, 31 Dec of this year")
    ap.add_argument("--epoch-date", default="31 Dec",
                    help="day and month of the start epoch, e.g. 07 Sep when the thrusting "
                         "descent lasts 250 days after the end of life on 31 Dec")
    ap.add_argument("--descent-days", type=float, default=0.0,
                    help="duration of the thrusting descent, added to every lifetime "
                         "because ESSB counts from the end of life")
    ap.add_argument("--years", type=int, default=12,
                    help="number of start epochs, one per year (a solar cycle is ~11 years)")
    ap.add_argument("--area", type=float, default=4.5, help="drag area, m^2")
    ap.add_argument("--mass", type=float, default=399.3, help="mass at the end of life, kg")
    ap.add_argument("--cd", type=float, default=2.2, help="Cd at the end of life (AD10 5.2)")
    ap.add_argument("--inc", type=float, default=97.47, help="inclination, deg")
    ap.add_argument("--percentile", default="Nominal",
                    help="Schatten error model: Nominal = 50th percentile (ESSB 6.2.g)")
    ap.add_argument("--max-days", type=float, default=7305, help="stop after this many days")
    ap.add_argument("--jobs", type=int, default=4, help="parallel GMAT runs")
    args = ap.parse_args()

    years = [args.first_year + k for k in range(args.years)]
    cases = [(h, y) for h in args.altitudes for y in years]
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        res = list(ex.map(lambda c: run_case(args, *c), cases))
    table = {c: r for c, r in zip(cases, res)}

    print("Orbit lifetime in years (start %s of the year, Cd %.1f, %.1f m^2, "
          "%s solar activity, plus %.0f days of descent); limit 5 years"
          % (args.epoch_date, args.cd, args.area, args.percentile, args.descent_days))
    print("%-10s" % "year" + "".join("%9.0f km" % h for h in args.altitudes))
    for y in years:
        print("%-10d" % y + "".join(
            "%12.2f" % ((table[(h, y)] + args.descent_days) / 365.25) for h in args.altitudes))
    print("%-10s" % "worst" + "".join(
        "%12.2f" % ((max(table[(h, y)] for y in years) + args.descent_days) / 365.25) for h in args.altitudes))
    print("%-10s" % "start 1" + "".join(
        "%12.2f" % ((table[(h, years[0])] + args.descent_days) / 365.25) for h in args.altitudes))
    ok = [h for h in args.altitudes
          if max(table[(h, y)] for y in years) + args.descent_days < 5 * 365.25]
    print("altitudes meeting the limit for every start epoch:", ok or "none")


if __name__ == "__main__":
    sys.exit(main())
