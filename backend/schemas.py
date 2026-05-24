"""schemas.py: Request and response schemas for the Random Houses API."""

__author__ = "Majd Jamal"

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """ Validated input features for a single house. Aliases map to the
    exact column names the model was trained on (which contain spaces). """

    gr_liv_area: float      = Field(alias = 'Gr Liv Area')
    lot_area: float         = Field(alias = 'Lot Area')
    overall_qual: int       = Field(alias = 'Overall Qual')
    overall_cond: int       = Field(alias = 'Overall Cond')
    year_built: int         = Field(alias = 'Year Built')
    year_remod_add: int     = Field(alias = 'Year Remod/Add')
    total_bsmt_sf: float    = Field(alias = 'Total Bsmt SF')
    garage_cars: int        = Field(alias = 'Garage Cars')
    garage_area: float      = Field(alias = 'Garage Area')
    full_bath: int          = Field(alias = 'Full Bath')
    totrms_abvgrd: int      = Field(alias = 'TotRms AbvGrd')
    fireplaces: int         = Field(alias = 'Fireplaces')

    neighborhood: str       = Field(alias = 'Neighborhood')
    house_style: str        = Field(alias = 'House Style')
    bldg_type: str          = Field(alias = 'Bldg Type')
    central_air: str        = Field(alias = 'Central Air')
    exter_qual: str         = Field(alias = 'Exter Qual')
    kitchen_qual: str       = Field(alias = 'Kitchen Qual')

    model_config = {'populate_by_name': True}


class PredictionResponse(BaseModel):
    request_id: str
    predicted_price: float
    model_version: str


class ModelInfo(BaseModel):
    model_version: str
    metrics: dict