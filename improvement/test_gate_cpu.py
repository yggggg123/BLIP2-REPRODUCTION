"""Quick real-Torch CPU checks; no LAVIS, model weights, downloads, or GPU use."""

import sys


def main():
    import torch
    from question_gate import QuestionChannelGate

    torch.set_num_threads(2)
    torch.manual_seed(1234)
    gate = QuestionChannelGate()
    assert sum(parameter.numel() for parameter in gate.parameters()) == 181056
    assert not any(name == "lavis" or name.startswith("lavis.") for name in sys.modules)

    queries = torch.randn(2, 32, 768)
    questions = torch.randn(2, 2048)
    target = torch.randn_like(queries)
    assert torch.equal(gate(queries, questions), queries)
    assert torch.equal(gate(queries, torch.zeros_like(questions)), queries)

    optimizer = torch.optim.SGD(gate.parameters(), lr=0.1)
    loss = (gate(queries, questions) - target).square().mean()
    loss.backward()
    assert gate.up.weight.grad.abs().sum().item() > 0
    assert gate.down.weight.grad.abs().sum().item() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss = (gate(queries, questions) - target).square().mean()
    loss.backward()
    assert gate.down.weight.grad.abs().sum().item() > 0

    # The same fixed question produces the same gate for the same visual input;
    # real, different questions can yield different gates once up learns.
    repeated_queries = queries[:1].repeat(2, 1, 1)
    constant_outputs = gate(repeated_queries, torch.zeros_like(questions))
    question_outputs = gate(repeated_queries, questions)
    assert torch.equal(constant_outputs[0], constant_outputs[1])
    assert not torch.equal(question_outputs[0], question_outputs[1])

    with torch.no_grad():
        gate.up.weight.normal_(std=4.0)
        gate.up.bias.normal_(std=4.0)
        scales = gate(torch.ones_like(queries), questions)
    assert scales.min().item() >= 0.75 - 1e-6
    assert scales.max().item() <= 1.25 + 1e-6
    assert torch.isfinite(scales).all()
    print("PASS: 181056 parameters; exact initial identity; first/second-step gradients; "
          "constant/question behavior; bounded scales [0.75,1.25]. CPU only, no LAVIS loaded.")


if __name__ == "__main__":
    main()
