import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

data = pd.read_json('data/cocktail_dataset.json')

# FULL PIPELINE
def pipeline(data: pd.DataFrame):

    def drop_unecessary_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
        return data.drop(columns=columns)


    def add_tag_vector_column(df: pd.DataFrame, tag_column: str) -> pd.DataFrame:
        # Extract unique tags from all rows in the tag column
        unique_tags = sorted(set(tag for tags_list in df[tag_column].dropna() for tag in tags_list))
        
        # Function to generate the vector for each row
        def tag_vector(tags):
            return [1 if tag in tags else 0 for tag in unique_tags]
        
        # Apply the function to create a new column with the tag vector
        df['tag_vector'] = df[tag_column].apply(lambda tags: tag_vector(tags) if tags else [0] * len(unique_tags))
        df.drop(columns= tag_column, inplace=True)
        
        return df  
    

    def add_ingredient_vector_column(df: pd.DataFrame, ingredient_column: str) -> pd.DataFrame:
        # Extract unique ingredient names from all rows in the ingredient column
        unique_ingredients = set(
            ingredient['name'] for ingredients_list in df[ingredient_column].dropna() 
            for ingredient in ingredients_list
        )
        unique_ingredients = sorted(unique_ingredients)  # Ensure consistent ordering
        
        # Function to generate the vector for each row
        def ingredient_vector(ingredients):
            ingredient_names = [ingredient['name'] for ingredient in ingredients]
            return [1 if ingredient in ingredient_names else 0 for ingredient in unique_ingredients]
        
        # Apply the function to create a new column with the ingredient vector
        df['ingredient_vector'] = df[ingredient_column].apply(lambda ingredients: ingredient_vector(ingredients) if ingredients else [0] * len(unique_ingredients))
        df.drop(columns= ingredient_column, inplace=True)
        return df


    def add_process_vector_column(df: pd.DataFrame, instruction_column: str) -> pd.DataFrame:
        # domain-based processes in bartending
        common_cocktail_processes = [
            'shake', 'stir', 'muddle', 'strain', 'blend', 'pour', 'build', 'layer',
            'rim', 'garnish', 'fill', 'squeeze', 'top', 'mix', 'flame', 'crush', 
            'dilute', 'press', 'double strain', 'dry shake', 'whip', 'float', 
            'swizzle', 'infuse', 'zest'
        ]

        # Function to extract processes from the instructions
        def extract_processes_from_instructions(instruction, processes_list):
            # Extract processes (simple case-insensitive match for known processes)
            processes_found = [process for process in processes_list if re.search(rf'\b{process.lower()}\b', instruction.lower())]
            return processes_found

        # Apply the function to extract processes
        df['processes_in_instructions'] = df[instruction_column].apply(lambda x: extract_processes_from_instructions(x, common_cocktail_processes))

        # Get unique processes found across all instructions
        unique_processes = sorted(set(process for sublist in df['processes_in_instructions'] for process in sublist))

        # Function to create a binary vector for processes
        def process_vector(processes, all_processes):
            return [1 if process in processes else 0 for process in all_processes]

        # Add the vectorized process column to the DataFrame
        df['process_vector'] = df['processes_in_instructions'].apply(lambda processes: process_vector(processes, unique_processes))
        df.drop(columns= [instruction_column, 'processes_in_instructions'], inplace=True)
        return df

    def normalize_vectors(df: pd.DataFrame, vector_columns: list) -> pd.DataFrame:
       
        def vector_norm(vectors: list) -> float:
            #Calculate the norm (magnitude) of the vector
            return np.linalg.norm(vectors)
        
        # Loop through each vector column and compute the norm
        for col in vector_columns:
            df[f'{col}_norm'] = df[col].apply(vector_norm)
        
        df = df.drop(columns=vector_columns)
        
        return df



    def OHE_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        # Initialize OneHotEncoder
        ohe = OneHotEncoder(sparse_output=False, drop = 'if_binary', handle_unknown = 'infrequent_if_exist')  
        
        # Apply OHE to the specified columns
        ohe_encoded = ohe.fit_transform(df[columns])
        
        # Convert the result to a DataFrame with appropriate column names
        ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(columns))
        
        # Drop the original columns that were encoded and concatenate the encoded columns
        df = df.drop(columns, axis=1)
        df = pd.concat([df, ohe_encoded_df], axis=1)
        
        return df


    def LE_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        # Dictionary to store the LabelEncoders for each column
        label_encoders = {}
        
        for col in columns:
            le = LabelEncoder()
            # Fit the LabelEncoder on the column and transform the values
            df[col] = le.fit_transform(df[col].astype(str))  # Converting to string to handle non-string types
            label_encoders[col] = le  
        
        return df

    
    # Applying functions:
    data = drop_unecessary_columns(data, ['imageUrl', 'createdAt', 'updatedAt', 'id'])
    data = add_tag_vector_column(data, 'tags')
    data = add_ingredient_vector_column(data, 'ingredients')
    data = add_process_vector_column(data, 'instructions')
    data = normalize_vectors(data, ['ingredient_vector', 'tag_vector', 'process_vector'])
    data = OHE_columns(data, ['category', 'alcoholic'])
    data = LE_columns(data, ['glass'])


    def apply_kmeans(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
        
        df = df.copy(deep=True)

        features_to_cluster = df.drop(columns=['name'])  # Drop 'name' column
        
        # Standardize the data for clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features_to_cluster)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_data)
        
        return df

    data = apply_kmeans(data, n_clusters = 6)
    
    return data[['name', 'cluster']]



result_df = pipeline(data)
result_df.to_csv('results/prediction1.csv', index=False)
print(result_df)

