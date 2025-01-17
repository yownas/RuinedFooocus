from gradio_client import Client

base_url = "http://localhost:7860"

search=""
max_results=25

client = Client(base_url)
result = client.predict(
        text=f"max:{max_results} {search}",
		api_name="/search"
)

for image in result:
    print(f"{base_url}/gradio_api/file/{image}")
