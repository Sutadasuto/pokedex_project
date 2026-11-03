import os
import requests
from concurrent.futures import ThreadPoolExecutor

def download_single_pokemon(i, base_folder, base_url):
    try:
        url = f"{base_url}{i}/"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return f"ID {i} no encontrado."
        
        data = response.json()
        base_name = data['name']
        # Usamos el ID i directamente para el número de la dex si es nacional
        pokedex_number = i 
        varieties = data["varieties"]          
            
        pokemon_path = os.path.join(base_folder, f"{pokedex_number:04d}_{base_name}")
        os.makedirs(pokemon_path, exist_ok=True) 

        # Sesión de requests para reutilizar la conexión TCP (más rápido)
        session = requests.Session()
        
        for variety in varieties:
            variety_url = variety["pokemon"]["url"]
            v_res = session.get(variety_url, timeout=10)
        
            if v_res.status_code != 200:
                continue
        
            v_data = v_res.json()
            v_name = v_data["name"]         
        
            sources = {
                'official-artwork': ['front_default', 'front_shiny'], 
                'home': ['front_default', 'front_female', 'front_shiny', 'front_shiny_female']
            }
            
            for key, types in sources.items():
                for img_type in types:
                    img_url = v_data['sprites']['other'][key].get(img_type)
                    if img_url:
                        img_content = session.get(img_url, timeout=10).content
                        filename = f"{pokedex_number:04d}_{v_name}_{key}_{img_type}.png"
                        with open(os.path.join(pokemon_path, filename), 'wb') as f:
                            f.write(img_content)
        return f"ID {i} ({base_name}) completado."
    except Exception as e:
        return f"Error en ID {i}: {e}"

def download_pokedex_fast(initial_dex_number=1, final_dex_number=1025, workers=10):
    base_folder = "pokedex_data"
    os.makedirs(base_folder, exist_ok=True)
    BASE_URL = "https://pokeapi.co/api/v2/pokemon-species/"

    print(f"Descargando Pokémon con {workers} hilos en paralelo...")

    # Ejecución en paralelo
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Mapeamos la función a la lista de IDs
        results = list(executor.map(
            lambda i: download_single_pokemon(i, base_folder, BASE_URL), 
            range(initial_dex_number, final_dex_number + 1)
        ))

    for res in results:
        if "Error" in res:
            print(res)

    print("\n¡Descarga masiva completada!")

if __name__ == "__main__":
    # Prueba con 10 o 20 workers. No subas demasiado o la API podría bloquearte temporalmente.
    download_pokedex_fast(workers=20)