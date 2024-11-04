import pygame
import digit_recognition as ai
import os

screen_size = (400, 400)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
brush_radius = 8

def drawing_window():
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Digit Recognition")
    screen.fill((255, 255, 255))
    drawing = False

    def predict_digit():
        pygame.image.save(screen, "drawn_digit.png")

        ai.recognize_digit("drawn_digit.png", False)

    def clear_screen():
        screen.fill(WHITE)

    pygame.init()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # C to clear
                    clear_screen()
                elif event.key == pygame.K_p:  # P to predict digit
                    predict_digit()

        # Draw:
        if drawing:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, BLACK, (mouse_x, mouse_y), brush_radius)

        # Update:
        pygame.display.flip()

while True:
    print("\nDigit Recognition App - option enter")
    print("1. Recognition on provided image path")
    print("2. Recognition on all images loaded in assets directory")
    print("3. Draw new digit")
    print("4. Display model performance statistics for currently loaded models")
    print("5. Display model summary for currently loaded models")
    print("6. Exit\n")

    choice = input("Enter your choice: ")

    if choice == '1':
        image_path = input("Enter image path: ")
        if os.path.isfile(image_path):
            ai.recognize_digit(image_path)
        else:
            print("Invalid image path")

    elif choice == '2':
        ai.recognize_digit_all_assets()

    elif choice == '3':
        drawing_window()

    elif choice == '4':
        ai.display_all_model_statistics()

    elif choice == '5':
        ai.display_all_model_summary()

    elif choice == '6':
        break

    else:
        print("Invalid choice")