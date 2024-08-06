from datetime import datetime

import numpy as np
import pandas as pd

from pypfate import Patch as patch
from pypfate import Clim

# Module will be moved to plantFATE package eventually and called from there

class PFatePatch:
    def __init__(self, param_file, acclim_forcing_file):
        self.patch = patch(str(param_file))
        self.time_unit_base = self.process_time_units()
        self.tcurrent = 0
        self.time = []
        self.swp = [] # Soil Water Potential Output for testing
        self.swc = [] # Soil Water Content for testing
        # self.smi = [] # Soil Moisture Index (?) for testing
        self.se = [] # Soil evaporation for testing
        self.trans = [] # Transpiration for testing
        self.acclimation_forcing = self.read_acclimation_file(acclim_forcing_file)

    def read_acclimation_file(self, file):
        df = pd.read_csv(file)
        alldates = df['date'].map(lambda x: datetime.strptime(x, "%Y-%m-%d") - self.time_unit_base)
        alldates = alldates.map(lambda x: x.days - 1)
        df['date_jul'] = alldates
        return df

    def runstep(
        self,
        soil_water_potential,
        vapour_pressure_deficit,
        photosynthetic_photon_flux_density,
        temperature,
        net_radiation,
        topsoil_volumetric_water_content,
        topsoil_moisture_wilting_point,
        topsoil_fieldcapacity,
        albedo
    ):
        self.patch.update_climate(
            368.9, # co2 - need to make it better
            temperature,
            vapour_pressure_deficit * 1000,
            photosynthetic_photon_flux_density,
            soil_water_potential,
            net_radiation
        )
        index_acclim = self.acclimation_forcing.index[self.acclimation_forcing['date_jul'] == self.tcurrent].tolist()
        self.patch.update_climate_acclim(self.tcurrent,
                                         368.9,
                                         self.acclimation_forcing.loc[index_acclim, 'temp.C.'],
                                         self.acclimation_forcing.loc[index_acclim, 'vpd'],
                                         self.calculate_photosynthetic_photon_flux_density(self.acclimation_forcing.loc[index_acclim, 'shortwave.W.m2.'], albedo),
                                         soil_water_potential)
        self.patch.simulate_to(self.tcurrent)
        trans = self.patch.props.fluxes.trans / 365
        potential_soil_evaporation = self.patch.props.fluxes.pe_soil
        soil_evaporation = self.calculate_actual_soil_evaporation(potential_soil_evaporation, topsoil_volumetric_water_content, topsoil_moisture_wilting_point, topsoil_fieldcapacity)
        evapotranspiration = trans + soil_evaporation
        self.trans.append(trans)
        self.se.append(soil_evaporation)
        # return evapotranspiration, soil_specific_depletion_1, soil_specific_depletion_2, soil_specific_depletion_3
        return trans, soil_evaporation, 0, 0, 0


    def first_step(
        self,
        tstart,
        soil_moisture_layer_1,  # ratio [0-1]
        soil_moisture_layer_2,  # ratio [0-1]
        soil_moisture_layer_3,  # ratio [0-1]
        soil_tickness_layer_1,  # m
        soil_tickness_layer_2,  # m
        soil_tickness_layer_3,  # m
        soil_moisture_wilting_point_1,  # ratio [0-1]print(type(time_unit_base))
        soil_moisture_wilting_point_2,  # ratio [0-1]
        soil_moisture_wilting_point_3,  # ratio [0-1]
        soil_moisture_field_capacity_1,  # ratio [0-1]
        soil_moisture_field_capacity_2,  # ratio [0-1]
        soil_moisture_field_capacity_3,  # ratio [0-1]
        temperature,  # degrees Celcius, mean temperature
        relative_humidity,  # percentage [0-100]
        shortwave_radiation_downwelling,  # W/m2, daily mean
        longwave_radiation_net,
        albedo
    ):
        (
            soil_water_potential,
            vapour_pressure_deficit,
            photosynthetic_photon_flux_density,
            temperature,
            net_radiation
        ) = self.get_plantFATE_input(
            soil_moisture_layer_1,  # ratio [0-1]
            soil_moisture_layer_2,  # ratio [0-1]
            soil_moisture_layer_3,  # ratio [0-1]
            soil_tickness_layer_1,  # m
            soil_tickness_layer_2,  # m
            soil_tickness_layer_3,  # m
            soil_moisture_wilting_point_1,  # ratio [0-1]
            soil_moisture_wilting_point_2,  # ratio [0-1]
            soil_moisture_wilting_point_3,  # ratio [0-1]
            soil_moisture_field_capacity_1,  # ratio [0-1]
            soil_moisture_field_capacity_2,  # ratio [0-1]
            soil_moisture_field_capacity_3,  # ratio [0-1]
            temperature,  # degrees Celcius, mean temperature
            relative_humidity,  # percentage [0-100]
            shortwave_radiation_downwelling,
            longwave_radiation_net,
            albedo)  # W/m2, daily mean

        newclim = Clim()
        newclim.tc = temperature #- 273.15  # C
        newclim.ppfd_max = photosynthetic_photon_flux_density * 4
        newclim.ppfd = photosynthetic_photon_flux_density
        newclim.vpd = vapour_pressure_deficit * 1000  # kPa -> Pa
        newclim.swp = soil_water_potential  # MPa


        # Convert time to proper units according to the time unit base
        datestart = datetime(tstart.year, tstart.month, tstart.day)
        datediff = datestart - self.time_unit_base
        datediff = datediff.days - 1



        self.swp.append(soil_water_potential)
        self.swc.append(soil_moisture_layer_1 + soil_moisture_layer_2 + soil_moisture_layer_3)
        # self.smi.append(self.calculate)
        self.se.append(0)
        self.trans.append(0)
        self.time.append(datestart)
        self.patch.init(datediff, datediff + 1000)
        self.tcurrent = datediff
        self.patch.update_climate(368.9,
                                  temperature,
                                  vapour_pressure_deficit * 1000,
                                  photosynthetic_photon_flux_density,
                                  soil_water_potential,
                                  net_radiation)
        index_acclim = self.acclimation_forcing.index[self.acclimation_forcing['date_jul'] == self.tcurrent].tolist()
        self.patch.update_climate_acclim(self.tcurrent,
                                         368.9,
                                         self.acclimation_forcing.loc[index_acclim, 'temp.C.'],
                                         self.acclimation_forcing.loc[index_acclim, 'vpd'],
                                         self.calculate_photosynthetic_photon_flux_density(self.acclimation_forcing.loc[index_acclim, 'shortwave.W.m2.'], albedo),
                                         soil_water_potential)

    def process_time_units(self):
        time_unit = self.patch.config.time_unit
        time_unit = time_unit.split()
        if time_unit[0] != 'days' or time_unit[1] != 'since':
            print("wrong plantfate unit; cwatm supports only daily timescale")
            return
        else:
            time_unit = time_unit[2].split("-")
            return datetime(int(time_unit[0]),
                            int(time_unit[1]),
                            int(time_unit[2]))

    # def read_acclimation_file(self):
    #     acclimation_file = self.patch.config.

    def simulate(self):
        self.patch.simulate()

    def calculate_soil_water_potential_MPa(
        self,
        soil_moisture,  # [m]
        soil_moisture_wilting_point,  # [m]
        soil_moisture_field_capacity,  # [m]
        soil_tickness,  # [m]
        wilting_point=-1500,  # kPa
        field_capacity=-33,  # kPa
    ):
        # https://doi.org/10.1016/B978-0-12-374460-9.00007-X (eq. 7.16)
        soil_moisture_fraction = soil_moisture / soil_tickness
        assert soil_moisture_fraction >= 0 and soil_moisture_fraction <= 1
        del soil_moisture
        soil_moisture_wilting_point_fraction = (
            soil_moisture_wilting_point / soil_tickness
        )
        assert (
            soil_moisture_wilting_point_fraction >= 0
            and soil_moisture_wilting_point_fraction <= 1
        )
        del soil_moisture_wilting_point
        soil_moisture_field_capacity_fraction = (
            soil_moisture_field_capacity / soil_tickness
        )
        assert (
            soil_moisture_field_capacity_fraction >= 0
            and soil_moisture_field_capacity_fraction <= 1
        )
        del soil_moisture_field_capacity

        n_potential = -(
            np.log(wilting_point / field_capacity)
            / np.log(
                soil_moisture_wilting_point_fraction
                / soil_moisture_field_capacity_fraction
            )
        )
        assert n_potential >= 0
        a_potential = (
            1.5 * 10**6 * soil_moisture_wilting_point_fraction**n_potential
        )
        assert a_potential >= 0
        soil_water_potential = -a_potential * soil_moisture_fraction ** (-n_potential)
        return soil_water_potential / 1_000_000  # Pa to MPa

    def calculate_vapour_pressure_deficit_kPa(self, temperature, relative_humidity):
        assert (
            temperature < 100
        )  # temperature is in Celsius. So on earth should be well below 100.
        assert (
            temperature > -100
        )  # temperature is in Celsius. So on earth should be well above -100.
        assert (
            relative_humidity >= 1 and relative_humidity <= 100
        )  # below 1 is so rare that it shouldn't be there at the resolutions of current climate models, and this catches errors with relative_humidity as a ratio [0-1].
        # https://soilwater.github.io/pynotes-agriscience/notebooks/vapor_pressure_deficit.html
        saturated_vapour_pressure = 0.611 * np.exp(
            (17.502 * temperature) / (temperature + 240.97)
        )  # kPa
        actual_vapour_pressure = (
            saturated_vapour_pressure * relative_humidity / 100
        )  # kPa
        vapour_pressure_deficit = saturated_vapour_pressure - actual_vapour_pressure
        return vapour_pressure_deficit

    def calculate_photosynthetic_photon_flux_density(self,
                                                     shortwave_radiation_downwelling,  # W/m2
                                                     albedo):  # [0-1]
        photosynthetic_photon_flux_density = shortwave_radiation_downwelling * (1-albedo) * 2.04 # umol/m2/s
        return photosynthetic_photon_flux_density

    def calculate_net_radiation(self, shortwave_radiation_downwelling, longwave_radiation_net, albedo):
        # TODO calculate this value
        net_radiation = shortwave_radiation_downwelling * (1-albedo) - longwave_radiation_net # W/m2
        return net_radiation

    def calculate_actual_soil_evaporation(self, potential_soil_evaporation, topsoil_water_content, topsoil_wilting_point, topsoil_fieldcap):
        topsoil_volumetric_content = (topsoil_water_content - topsoil_wilting_point)/(topsoil_fieldcap - topsoil_wilting_point)
        actual_soil_evaporation = potential_soil_evaporation * topsoil_volumetric_content
        return actual_soil_evaporation

    def get_plantFATE_input(
        self,
        soil_moisture_layer_1,  # m
        soil_moisture_layer_2,  # m
        soil_moisture_layer_3,  # m
        soil_tickness_layer_1,  # m
        soil_tickness_layer_2,  # m
        soil_tickness_layer_3,  # m
        soil_moisture_wilting_point_1,  # m
        soil_moisture_wilting_point_2,  # m
        soil_moisture_wilting_point_3,  # m
        soil_moisture_field_capacity_1,  # m
        soil_moisture_field_capacity_2,  # m
        soil_moisture_field_capacity_3,  # m
        temperature,  # degrees Celcius, mean temperature
        relative_humidity,  # percentage [0-100]
        shortwave_radiation_downwelling,  # W/m2, daily mean
        longwave_radiation_net,
        albedo
    ):
        assert (
            temperature < 100
        )  # temperature is in Celsius. So on earth should be well below 100.
        assert relative_humidity >= 0 and relative_humidity <= 100

        soil_water_potential = self.calculate_soil_water_potential_MPa(
            soil_moisture_layer_1 + soil_moisture_layer_2 + soil_moisture_layer_3,
            soil_moisture_wilting_point_1
            + soil_moisture_wilting_point_2
            + soil_moisture_wilting_point_3,
            soil_moisture_field_capacity_1
            + soil_moisture_field_capacity_2
            + soil_moisture_field_capacity_3,
            soil_tickness_layer_1 + soil_tickness_layer_2 + soil_tickness_layer_3,
        )

        vapour_pressure_deficit = self.calculate_vapour_pressure_deficit_kPa(
            temperature, relative_humidity
        )

        photosynthetic_photon_flux_density = (
            self.calculate_photosynthetic_photon_flux_density(shortwave_radiation_downwelling, albedo)
        )

        net_radiation = self.calculate_net_radiation(shortwave_radiation_downwelling, longwave_radiation_net, albedo)

        return (
            soil_water_potential,
            vapour_pressure_deficit,
            photosynthetic_photon_flux_density,
            temperature,
            net_radiation
        )

    def step(
        self,
        curr_time,
        soil_moisture_layer_1,  # ratio [0-1]
        soil_moisture_layer_2,  # ratio [0-1]
        soil_moisture_layer_3,  # ratio [0-1]
        soil_tickness_layer_1,  # m
        soil_tickness_layer_2,  # m
        soil_tickness_layer_3,  # m
        soil_moisture_wilting_point_1,  # ratio [0-1]
        soil_moisture_wilting_point_2,  # ratio [0-1]
        soil_moisture_wilting_point_3,  # ratio [0-1]
        soil_moisture_field_capacity_1,  # ratio [0-1]
        soil_moisture_field_capacity_2,  # ratio [0-1]
        soil_moisture_field_capacity_3,  # ratio [0-1]
        temperature,  # degrees Celcius, mean temperature
        relative_humidity,  # percentage [0-100]
        shortwave_radiation_downwelling,  # W/m2, daily mean
        longwave_radiation_net,
        albedo
    ):
        (
            soil_water_potential,
            vapour_pressure_deficit,
            photosynthetic_photon_flux_density,
            temperature,
            net_radiation
        ) = self.get_plantFATE_input(
            soil_moisture_layer_1,  # ratio [0-1]
            soil_moisture_layer_2,  # ratio [0-1]
            soil_moisture_layer_3,  # ratio [0-1]
            soil_tickness_layer_1,  # m
            soil_tickness_layer_2,  # m
            soil_tickness_layer_3,  # m
            soil_moisture_wilting_point_1,  # ratio [0-1]
            soil_moisture_wilting_point_2,  # ratio [0-1]
            soil_moisture_wilting_point_3,  # ratio [0-1]
            soil_moisture_field_capacity_1,  # ratio [0-1]
            soil_moisture_field_capacity_2,  # ratio [0-1]
            soil_moisture_field_capacity_3,  # ratio [0-1]
            temperature,  # degrees Celcius, mean temperature
            relative_humidity,  # percentage [0-100]
            shortwave_radiation_downwelling,  # W/m2, daily mean
            longwave_radiation_net,
            albedo
        )

        curr_time_dt = datetime(curr_time.year, curr_time.month, curr_time.day)
        timediff = curr_time_dt - self.time_unit_base
        self.tcurrent = timediff.days - 1


        (
            transpiration,
            soil_evaporation,
            soil_specific_depletion_1,
            soil_specific_depletion_2,
            soil_specific_depletion_3,
        ) = self.runstep(
            soil_water_potential,
            vapour_pressure_deficit,
            photosynthetic_photon_flux_density,
            temperature,
            net_radiation,
            soil_moisture_layer_1,
            soil_moisture_wilting_point_1,
            soil_moisture_field_capacity_1,
            albedo
        )

        self.swp.append(soil_water_potential)
        self.swc.append(soil_moisture_layer_1 + soil_moisture_layer_2 + soil_moisture_layer_3)
        self.time.append(curr_time_dt)


        soil_specific_depletion_1 = (
            np.nan
        )  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion
        soil_specific_depletion_2 = (
            np.nan
        )  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion
        soil_specific_depletion_3 = (
            np.nan
        )  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion

        if curr_time.month == 12 and curr_time.day == 31:
            df = pd.DataFrame(data={'Time': self.time,
                                    'SWP': self.swp,
                                    'SWC': self.swc,
                                    'Trans': self.trans})
            # write the DataFrame to a CSV file
            df.to_csv('Soil_Water.csv')

        transpiration = transpiration / 1000  # kg H2O/m2/day to m/day

        return (
            transpiration,
            soil_evaporation,
            soil_specific_depletion_1,
            soil_specific_depletion_2,
            soil_specific_depletion_3,
        )

    def finalize(self):

        # create a Pandas DataFrame from the dictionary
        if(len(self.swp) > 0):
            df = pd.DataFrame(data={'Time': list(range(1, len(self.swp) + 1)),
                                    'SWP': self.swp,
                                    'SWC': self.swc,
                                    'SE': self.se,
                                    'trans': self.trans,
                                    })

            # write the DataFrame to a CSV file
            df.to_csv('SWP.csv')

        print("closing patch")
        self.patch.close()

    @property
    def n_individuals(self):
        return sum(self.patch.props.structure.n_ind_vec)

    @property
    def biomass(self):
        return sum(self.patch.cwm.biomass_vec)  # kgC / m2
