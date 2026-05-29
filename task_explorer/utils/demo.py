from traj_to_kg import TrajectoryToNeo4jImporter, find_all_task_folders
import config
import requests
import json
import config


def main():
    # response = requests.get(
    #     url="https://openrouter.ai/api/v1/key",
    #     headers={
    #         "Authorization": f"Bearer {config.LLM_API_KEY}"
    #     }
    # )
    # print(json.dumps(response.json(), indent=2))

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "anthropic/claude-sonnet-4",
        "messages": [
            {
                "role": "user",
                "content": "If you built the world's tallest skyscraper, what would you name it?"
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    print(response.json())

if __name__ == "__main__":
    main()