from sglang.simulator.srt.entrypoints.engine import EngineSimulator


def main():
    simulator_args = {
        "gpu_size_actual": 1,
    }

    llm = EngineSimulator(
        model_path="/data/llama-3-8b-instruct/",
        tp_size=2,
        base_gpu_id=0,
        disable_cuda_graph=True,

        # simulation
        simulator_args=simulator_args,
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()
