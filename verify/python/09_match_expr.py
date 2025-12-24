# Test: Match expression
def grade(score):
    if score == 100:
        return 5
    elif score >= 80:
        return 4
    elif score >= 60:
        return 3
    elif score >= 40:
        return 2
    else:
        return 1

def main():
    return grade(100) + grade(85) + grade(65) + grade(45) + grade(20)

print(main())
