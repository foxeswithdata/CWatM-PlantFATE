# -------------------------------------------------------------------------
# Name:        Capillar Rise module
# Purpose:
#
# Author:      PB
#
# Created:     20/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

from cwatm.management_modules.data_handling import *


class capillarRise(object):
    """
    CAPILLAR RISE
    calculate cell fraction influenced by capillary rise
    **Functions**
    """

    def __init__(self, model):
        self.var = model.var
        self.model = model

    def dynamic(self):
        """
        Dynamic part of the capillar Rise module
        calculate cell fraction influenced by capillary rise
        depending on appr. height of groundwater and relative elevation of grid cell

        :return: capRiseFrac = cell fraction influenced by capillary rise
        """

        if checkOption('CapillarRise') and not (self.var.modflow):

            # approximate height of groundwater table and corresponding reach of cell under influence of capillary rise
            dzGroundwater = self.var.storGroundwater / self.var.specificYield + self.var.maxGWCapRise
            CRFRAC = np.minimum(1.0, 1.0 - (self.var.dzRel[11] - dzGroundwater) * 0.1 / np.maximum(1e-3,
                  self.var.dzRel[11] - self.var.dzRel[10]))

            #       10  9   8   7   6   5    4   3   2  1    0
            vvv = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01]
            i = 10
            for vv in vvv:
                if i > 0:
                    h = vv-(self.var.dzRel[i]-dzGroundwater) * 0.1 / np.maximum(1e-3,self.var.dzRel[i]-self.var.dzRel[i-1])
                else:
                    h = vv-(self.var.dzRel[i]-dzGroundwater) * 0.1 / np.maximum(1e-3,self.var.dzRel[i])
                CRFRAC = np.where(dzGroundwater < self.var.dzRel[i], h , CRFRAC)
                i = i - 1

            self.var.capRiseFrac = np.maximum(0.0, np.minimum(1.0, CRFRAC))
        else:
            self.var.capRiseFrac = 0.


