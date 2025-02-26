import streamlit as st
import requests
import json
import pandas as pd


def get_default_values():
    """Return a dictionary of default values for all features"""
    return {
        'Id': 1,
        'MSSubClass': 60,  # 2-STORY 1946 & NEWER
        'MSZoning': 'RL',  # Residential Low Density
        'LotFrontage': 65.0,
        'LotArea': 8450,
        'Street': 'Pave',
        'Alley': 'NA',
        'LotShape': 'Reg',
        'LandContour': 'Lvl',
        'Utilities': 'AllPub',
        'LotConfig': 'Inside',
        'LandSlope': 'Gtl',
        'Neighborhood': 'CollgCr',
        'Condition1': 'Norm',
        'Condition2': 'Norm',
        'BldgType': '1Fam',
        'HouseStyle': '2Story',
        'OverallQual': 7,
        'OverallCond': 5,
        'YearBuilt': 2003,
        'YearRemodAdd': 2003,
        'RoofStyle': 'Gable',
        'RoofMatl': 'CompShg',
        'Exterior1st': 'VinylSd',
        'Exterior2nd': 'VinylSd',
        'MasVnrType': 'BrkFace',
        'MasVnrArea': 196.0,
        'ExterQual': 'Gd',
        'ExterCond': 'TA',
        'Foundation': 'PConc',
        'BsmtQual': 'Gd',
        'BsmtCond': 'TA',
        'BsmtExposure': 'No',
        'BsmtFinType1': 'GLQ',
        'BsmtFinSF1': 706,
        'BsmtFinType2': 'Unf',
        'BsmtFinSF2': 0,
        'BsmtUnfSF': 150,
        'TotalBsmtSF': 856,
        'Heating': 'GasA',
        'HeatingQC': 'Ex',
        'CentralAir': 'Y',
        'Electrical': 'SBrkr',
        '1stFlrSF': 856,
        '2ndFlrSF': 854,
        'LowQualFinSF': 0,
        'GrLivArea': 1710,
        'BsmtFullBath': 1,
        'BsmtHalfBath': 0,
        'FullBath': 2,
        'HalfBath': 1,
        'BedroomAbvGr': 3,
        'KitchenAbvGr': 1,
        'KitchenQual': 'Gd',
        'TotRmsAbvGrd': 8,
        'Functional': 'Typ',
        'Fireplaces': 0,
        'FireplaceQu': 'NA',
        'GarageType': 'Attchd',
        'GarageYrBlt': 2003.0,
        'GarageFinish': 'RFn',
        'GarageCars': 2,
        'GarageArea': 548,
        'GarageQual': 'TA',
        'GarageCond': 'TA',
        'PavedDrive': 'Y',
        'WoodDeckSF': 0,
        'OpenPorchSF': 61,
        'EnclosedPorch': 0,
        '3SsnPorch': 0,
        'ScreenPorch': 0,
        'PoolArea': 0,
        'PoolQC': 'NA',
        'Fence': 'NA',
        'MiscFeature': 'NA',
        'MiscVal': 0,
        'MoSold': 2,
        'YrSold': 2008,
        'SaleType': 'WD',
        'SaleCondition': 'Normal'

    }


def initialize_session_state():
    """Initialize session state variables"""
    if 'features' not in st.session_state:
        st.session_state.features = get_default_values()


def main():
    st.title("House Price Prediction")
    st.write("Enter house features to predict the price")

    # Initialize session state
    initialize_session_state()

    # Create tabs for different feature categories
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Info", "Interior Features", "Exterior Features", "Other Features"])

    with tab1:
        st.subheader("Basic Information")
        # Use session state for widget values
        if 'Id' not in st.session_state:
            st.session_state.Id = 1

        st.session_state.features.update({
            'Id': st.number_input('House ID',
                                  min_value=1,
                                  max_value=10000,
                                  value=st.session_state.Id,
                                  key='house_id',
                                  help="Unique identifier for the house"
                                  ),
            'MSSubClass': st.selectbox('Building Class',
                                       options=[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
                                       index=5,
                                       key='building_class',
                                       help="The building class"
                                       ),
            'MSZoning': st.selectbox('Zoning Classification',
                                     options=['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
                                     index=5,
                                     key='zoning'
                                     ),
            'LotArea': st.number_input('Lot Size (sq ft)',
                                       min_value=1000,
                                       max_value=100000,
                                       value=st.session_state.features['LotArea'],
                                       key='lot_area'
                                       ),
            'YearBuilt': st.number_input('Year Built',
                                         min_value=1800,
                                         max_value=2024,
                                         value=st.session_state.features['YearBuilt'],
                                         key='year_built'
                                         ),
            'YearRemodAdd': st.number_input('Year Remodeled',
                                            min_value=1800,
                                            max_value=2024,
                                            value=st.session_state.features['YearRemodAdd'],
                                            key='year_remod'
                                            )
        })

    with tab2:
        st.subheader("Interior Features")
        st.session_state.features.update({
            'OverallQual': st.slider('Overall Quality', 1, 10,
                                     value=st.session_state.features['OverallQual'],
                                     key='overall_qual'),
            'OverallCond': st.slider('Overall Condition', 1, 10,
                                     value=st.session_state.features['OverallCond'],
                                     key='overall_cond'),
            'TotalBsmtSF': st.number_input('Total Basement Area (sq ft)',
                                           min_value=0,
                                           max_value=6000,
                                           value=st.session_state.features['TotalBsmtSF'],
                                           key='total_bsmt'
                                           ),
            'GrLivArea': st.number_input('Above Ground Living Area (sq ft)',
                                         min_value=300,
                                         max_value=10000,
                                         value=st.session_state.features['GrLivArea'],
                                         key='gr_liv_area'
                                         ),
            'FullBath': st.number_input('Full Bathrooms',
                                        min_value=0,
                                        max_value=5,
                                        value=st.session_state.features['FullBath'],
                                        key='full_bath'
                                        ),
            'BedroomAbvGr': st.number_input('Bedrooms',
                                            min_value=0,
                                            max_value=10,
                                            value=st.session_state.features['BedroomAbvGr'],
                                            key='bedrooms'
                                            )
        })

    with tab3:
        st.subheader("Exterior Features")
        st.session_state.features.update({
            'GarageType': st.selectbox('Garage Type',
                                       options=['Attchd', 'Detachd', 'BuiltIn', 'CarPort', 'NA'],
                                       index=0,
                                       key='garage_type'
                                       ),
            'GarageArea': st.number_input('Garage Area (sq ft)',
                                          min_value=0,
                                          max_value=2000,
                                          value=st.session_state.features['GarageArea'],
                                          key='garage_area'
                                          ),
            'WoodDeckSF': st.number_input('Wood Deck Area (sq ft)',
                                          min_value=0,
                                          max_value=1000,
                                          value=st.session_state.features['WoodDeckSF'],
                                          key='wood_deck'
                                          ),
            'OpenPorchSF': st.number_input('Open Porch Area (sq ft)',
                                           min_value=0,
                                           max_value=1000,
                                           value=st.session_state.features['OpenPorchSF'],
                                           key='open_porch'
                                           )
        })

    if st.button("Predict Price", key='predict_button'):

        try:
            #st.write("Sending features:", st.session_state.features)

            response = requests.post(
                "http://localhost:8000/predict",
                json={"features": st.session_state.features},
                headers={"Content-Type": "application/json"}
            )
            #st.write("raw response", response.text)

            if response.status_code == 200:
                result = response.json()
                #st.write("JSON response:", result)  # Debug the response

                if 'predicted_price' in result:
                    predicted_price = result['predicted_price']
                    st.success(f"Predicted House Price: ${predicted_price:,.2f}")
                else:
                    st.error(f"Response is missing 'predicted_price' key. Response: {result}")

                st.write("\nKey features of this house:")
                st.write(f"- Living Area: {st.session_state.features['GrLivArea']} sq ft")
                st.write(f"- Overall Quality: {st.session_state.features['OverallQual']}/10")
                st.write(f"- Year Built: {st.session_state.features['YearBuilt']}")
                st.write(f"- Total Bathrooms: {st.session_state.features['FullBath']}")
            else:
                st.error(f"Error: {response.text}")


        except Exception as e:

            st.error(f"Error making prediction: {str(e)}")

            st.write("Full error:", e)  # Show the full error


if __name__ == "__main__":
    main()