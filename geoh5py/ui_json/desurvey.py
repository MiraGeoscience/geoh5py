#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.


def updateIntervalInfo(self):
    intervals.clear()

    if self.isCartesianOnly():
        return

    full_surveys = self.getSurveysForFullPath()

    for i in range(len(full_surveys)):
        survey = full_surveys[i]
        # adjust az for grid azimuth and geo->cart coord system
        az = 90 - survey.azim
        intervals[survey.depth] = IBHInterval(survey.depth, az, survey.dip)

    rd = az = di = sn = cs = dp0 = dp1 = cx0 = cy0 = cz0 = cx1 = cy1 = cz1 = vx = vy = (
        vz
    ) = vr = ux = uy = uz = ur = 0.0
    scp = ddp = alpha = x0 = y0 = z0 = x1 = y1 = z1 = 0.0
    tdepth = prevdepth = 0.0
    it_bh = iter(intervals.items())

    if intervals:
        it_bh = iter(intervals.items())
        key, value = next(it_bh)
        tdepth = prevdepth = dp1 = abs(key)
        az = DegToRad(value.azim)
        di = DegToRad(value.dip)

        value.xh = collar.position.x
        value.yh = collar.position.y
        value.zh = collar.position.z
    else:
        tdepth = prevdepth = dp1 = 0
        az = 0.0
        di = DegToRad(-90.0)

    cs = math.cos(di)
    sn = math.sin(di)
    cx1 = cs * math.cos(az)
    cy1 = cs * math.sin(az)
    cz1 = sn
    if len(intervals) > 1:
        it_bh = iter(intervals.items())
        next(it_bh)
        for key, value in it_bh:
            tdepth = key
            az = DegToRad(value.azim)
            di = DegToRad(value.dip)
            dp0 = dp1
            dp1 = abs(tdepth)
            ddp = dp1 - dp0
            if ddp == 0.0:
                ddp = 1.0
            cx0 = cx1
            cy0 = cy1
            cz0 = cz1
            cs = math.cos(di)
            sn = math.sin(di)
            cx1 = cs * math.cos(az)
            cy1 = cs * math.sin(az)
            cz1 = sn
            # v = cross product between c0 and c1
            vx = cy0 * cz1 - cz0 * cy1
            vy = cz0 * cx1 - cx0 * cz1
            vz = cx0 * cy1 - cy0 * cx1
            vr = math.sqrt(vx * vx + vy * vy + vz * vz)
            # scp = dot product between c0 and c1
            scp = cx0 * cx1 + cy0 * cy1 + cz0 * cz1
            ux = cx1 - scp * cx0
            uy = cy1 - scp * cy0
            uz = cz1 - scp * cz0
            ur = math.sqrt(ux * ux + uy * uy + uz * uz)
            if ur == 0.0:
                ur = INFINITE_RADIUS
            ux = ux / ur
            uy = uy / ur
            uz = uz / ur

            # scp/vr = cos(gamma)/sin(gamma) = cot(gamma) with gamma the angle between c0 and c1.
            # arccot(x) = pi/2 - arctan(x)
            # (https://en.wikipedia.org/wiki/Inverse_trigonometric_functions) so alpha = abs(pi/2 -
            # arctan(scp/vr)) = abs(arccot(scp/vr)) = abs(arccot(cot(gamma))) = abs(gamma) alpha is
            # the positive angle between c0 and c1.
            alpha = abs(0.5 * math.pi - qAtan2(scp, vr))
            # {should range from 0 to pi}
            if alpha != 0.0:
                # length of an arc of circle (here ddp) = radius * angle,
                # so we approximate the path between 2 stations by an arc of circle.
                rd = ddp / alpha
            else:
                rd = ddp * INFINITE_RADIUS  # {flag infinite radius}
                alpha = ddp / rd
            intervals[prevdepth].rad = rd
            intervals[prevdepth].uux = ux
            intervals[prevdepth].uuy = uy
            intervals[prevdepth].uuz = uz
            intervals[prevdepth].ccx = cx0
            intervals[prevdepth].ccy = cy0
            intervals[prevdepth].ccz = cz0
            x0 = intervals[prevdepth].xh
            y0 = intervals[prevdepth].yh
            z0 = intervals[prevdepth].zh
            sn = math.sin(alpha)
            cs = 1.0 - math.cos(alpha)
            x1 = x0 + rd * (cx0 * sn + ux * cs)
            y1 = y0 + rd * (cy0 * sn + uy * cs)
            z1 = z0 + rd * (cz0 * sn + uz * cs)
            intervals[tdepth].xh = x1
            intervals[tdepth].yh = y1
            intervals[tdepth].zh = z1
            prevdepth = tdepth

        intervals[tdepth].uux = 0.0
        intervals[tdepth].uuy = 0.0
        intervals[tdepth].uuz = 0.0
        intervals[tdepth].rad = INFINITE_RADIUS
        intervals[tdepth].ccx = cx1
        intervals[tdepth].ccy = cy1
        intervals[tdepth].ccz = cz1
    else:
        if 0.0 not in intervals:
            intervals[0.0] = IBHInterval(tdepth, az, di)
        intervals[0.0].rad = INFINITE_RADIUS
        intervals[0.0].ccx = cx1
        intervals[0.0].ccy = cy1
        intervals[0.0].ccz = cz1
        intervals[0.0].uux = 0.0
        intervals[0.0].uuy = 0.0
        intervals[0.0].uuz = 0.0
        intervals[0.0].xh = collar.position.x
        intervals[0.0].yh = collar.position.y
        intervals[0.0].zh = collar.position.z
