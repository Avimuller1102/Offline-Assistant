import typer
from rich import print

from echoshield.engine import EchoShield, EchoConfig

app = typer.Typer(help="EchoShield Offline Assistant CLI")

@app.command()
def chat(
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    rpm: int = typer.Option(60, help="Rate limit RPM"),
    kb_path: str = typer.Option("echoshield_kb.json", help="Path to BM25 KB json"),
):
    """
    Start an interactive offline chat session with EchoShield.
    """
    print(f"[bold green]Starting EchoShield (temp={temperature}, rpm={rpm})[/bold green]")
    print("[italic]Loading local models. This may take a few seconds...[/italic]")
    
    cfg = EchoConfig(
        temperature=temperature,
        rpm_limit=rpm,
        kb_path=kb_path
    )
    
    try:
        shield = EchoShield(cfg)
    except Exception as e:
        print(f"[bold red]Error loading models:[/bold red] {e}")
        raise typer.Exit(1)
        
    print("[bold blue]EchoShield Ready! Type 'exit' to quit.[/bold blue]")
    print("-" * 50)
    
    while True:
        try:
            user_input = typer.prompt("You")
            if user_input.lower() in ("exit", "quit"):
                break
                
            shield.memory.add("User", user_input)
            response, _ = shield._consensus(user_input)
            shield.memory.add("Bot", response)
            
            print(f"[bold green]EchoShield:[/bold green] {response}")
            print("-" * 50)
            
        except typer.Abort:
            break
        except Exception as e:
            print(f"[bold red]Error:[/bold red] {e}")

@app.command()
def info():
    """
    Print information about EchoShield configuration.
    """
    print("[bold]EchoShield v2 Offline Assistant[/bold]")
    print("This assistant runs completely offline using local huggingface models.")

if __name__ == "__main__":
    app()
