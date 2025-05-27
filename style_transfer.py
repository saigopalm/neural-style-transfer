from models.model import get_style_model
from utils.optimizer import get_optimizer

def run_style_transfer(cnn, content_img, style_img, input_img,
                       style_weight=1e6, content_weight=1, num_steps=300,
                       style_layers=None, content_layers=None):

    model, style_losses, content_losses = get_style_model(
        cnn, 0, 1, style_img, content_img, style_layers, content_layers)

    model.eval()

    optimizer = get_optimizer(input_img)

    print('Optimizing...')
    step = [0]

    while step[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)

            loss = style_score * style_weight + content_score * content_weight
            loss.backward()

            if step[0] % 10 == 0:
                print(f"Step {step[0]} - Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")

            step[0] += 1
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img