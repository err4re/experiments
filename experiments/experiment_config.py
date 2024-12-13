from instruments.configs.znb_config import ZnbLinConfig, ZnbSegmConfig, ZnbExtTrigOutConfig, ZnbCWConfig
from instruments.configs.yoko7651_config import YokoCurrSweepConfig, YokoVoltSweepConfig
from instruments.configs.anapico_config import AnaFreqSweepConfig, AnaExtTrigInConfig

from typing import Union, Optional
from dataclasses import dataclass


#Base configuration class, e.g. for FluxMapConfig or TwoToneConfig
class ExperimentConfig:
    pass


@dataclass
class FluxMapConfig(ExperimentConfig):
    yoko: Union[YokoCurrSweepConfig, YokoVoltSweepConfig]
    vna: Union[ZnbLinConfig, ZnbSegmConfig]


    def __post_init__(self):
        if self.yoko is None:
            raise ValueError("yoko config must be set")
        if self.vna is None:
            raise ValueError("vna config must be set")
        

@dataclass
class TwoToneConfig(ExperimentConfig):
    yoko: Union[YokoCurrSweepConfig, YokoVoltSweepConfig]
    vna_f1_calib: Union[ZnbLinConfig, ZnbSegmConfig]
    vna_f1: Union[ZnbLinConfig, ZnbSegmConfig, ZnbCWConfig]
    vna_trigger: ZnbExtTrigOutConfig
    ana_f2: AnaFreqSweepConfig
    ana_trigger: AnaExtTrigInConfig
    tracking_parameters: Optional[dict] = None 
    f2_to_span: Optional[callable] = None
    f2_to_power: Optional[callable] = None


    def __post_init__(self):
        if self.yoko is None:
            raise ValueError("yoko must be set")
        if self.vna_f1_calib is None:
            raise ValueError("vna_f1_calib must be set")
        if self.vna_f1 is None:
            raise ValueError("vna_f1 must be set")
        if self.vna_trigger is None:
            raise ValueError("vna_trigger must be set")
        if self.ana_f2 is None:
            raise ValueError("ana_f2 must be set")
        if self.ana_trigger is None:
            raise ValueError("ana_trigger must be set")