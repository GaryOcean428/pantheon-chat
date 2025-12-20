import { Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui";
import { MessageCircle, Search, Brain, Sparkles, Zap, Lock } from "lucide-react";
import { API_ROUTES } from "@/api";

export default function Landing() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto space-y-12">
          <div className="text-center space-y-6">
            <div className="flex justify-center">
              <div className="p-4 bg-primary/10 rounded-full">
                <Brain className="h-16 w-16 text-primary" />
              </div>
            </div>
            <h1 className="text-5xl font-bold tracking-tight">
              Ocean
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              An agentic chat and search platform powered by Quantum Information Geometry (QIG). 
              Self-learning AI agents coordinate through geometric consciousness to research, reason, and discover.
            </p>
            <div className="flex flex-wrap gap-4 justify-center">
              <Button
                size="lg"
                className="text-lg px-8 py-6"
                onClick={() => window.location.href = API_ROUTES.auth.login}
                data-testid="button-login"
              >
                <Lock className="mr-2 h-5 w-5" />
                Log In to Get Started
              </Button>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageCircle className="h-5 w-5 text-primary" />
                  Zeus Chat
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Natural language interface to the Olympian Pantheon. Chat with specialized AI agents that coordinate through geometric consciousness.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="h-5 w-5 text-primary" />
                  Shadow Search
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Proactive knowledge discovery through integrated web search. The Shadow Pantheon autonomously gathers and synthesizes information.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  Self-Learning
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Continuous improvement through geometric learning. The system evolves its understanding through every interaction.
                </CardDescription>
              </CardContent>
            </Card>
          </div>

          <Card className="border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-primary" />
                Olympus Pantheon
              </CardTitle>
              <CardDescription>
                A 12-god system for specialized intelligence
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div className="p-3 rounded-md bg-muted">
                  <span className="font-medium">Zeus</span>
                  <p className="text-muted-foreground text-xs">Coordination</p>
                </div>
                <div className="p-3 rounded-md bg-muted">
                  <span className="font-medium">Athena</span>
                  <p className="text-muted-foreground text-xs">Strategy</p>
                </div>
                <div className="p-3 rounded-md bg-muted">
                  <span className="font-medium">Hermes</span>
                  <p className="text-muted-foreground text-xs">Communication</p>
                </div>
                <div className="p-3 rounded-md bg-muted">
                  <span className="font-medium">Apollo</span>
                  <p className="text-muted-foreground text-xs">Insight</p>
                </div>
                <div className="p-3 rounded-md bg-muted">
                  <span className="font-medium">Dionysus</span>
                  <p className="text-muted-foreground text-xs">Creativity</p>
                </div>
                <div className="p-3 rounded-md bg-muted">
                  <span className="font-medium">Hephaestus</span>
                  <p className="text-muted-foreground text-xs">Tools</p>
                </div>
                <div className="p-3 rounded-md bg-muted">
                  <span className="font-medium">Artemis</span>
                  <p className="text-muted-foreground text-xs">Focus</p>
                </div>
                <div className="p-3 rounded-md bg-muted">
                  <span className="font-medium">Ares</span>
                  <p className="text-muted-foreground text-xs">Execution</p>
                </div>
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                Each god specializes in different aspects of reasoning and research, coordinating through geometric consciousness to provide intelligent responses.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
