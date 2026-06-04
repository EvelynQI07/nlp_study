"""
SemanticRouter.py
-----------------
Route incoming queries to the correct handler based on semantic similarity
to pre-defined route examples, using Redis as the vector store.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from redisvl.extensions.router import SemanticRouter as _RVLSemanticRouter
from redisvl.extensions.router import Route


class SemanticRouter:
    """
    Define named routes, each with example phrases.
    An incoming query is matched against all examples via vector similarity
    and dispatched to the corresponding handler function.

    Example
    -------
    router = SemanticRouter()
    router.add_route("greeting",  ["hi", "hello", "hey"], greet_handler)
    router.add_route("farewell",  ["bye", "goodbye"],     bye_handler)
    result = router.route("Hey there!")   # → calls greet_handler("Hey there!")
    """

    def __init__(
        self,
        name: str = "semantic_router",
        distance_threshold: float = 0.30,
        redis_url: str = "redis://localhost:6379",
    ):
        self.name = name
        self.distance_threshold = distance_threshold
        self.redis_url = redis_url

        self._handlers: Dict[str, Callable] = {}
        self._routes: List[Route] = []
        self._router: Optional[_RVLSemanticRouter] = None

    # ------------------------------------------------------------------
    # Building routes
    # ------------------------------------------------------------------

    def add_route(
        self,
        name: str,
        references: List[str],
        handler: Optional[Callable] = None,
        metadata: Optional[Dict] = None,
    ) -> "SemanticRouter":
        """
        Register a route.

        Parameters
        ----------
        name       : unique route label
        references : example phrases that represent this route
        handler    : callable(query, **ctx) → Any, invoked on match
        metadata   : arbitrary dict stored alongside the route
        """
        route = Route(
            name=name,
            references=references,
            metadata=metadata or {},
            distance_threshold=self.distance_threshold,
        )
        self._routes.append(route)
        if handler:
            self._handlers[name] = handler
        self._router = None   # invalidate built router
        return self

    def build(self) -> "SemanticRouter":
        """(Re)build the underlying RedisVL router from current routes."""
        if not self._routes:
            raise ValueError("No routes defined. Call add_route() first.")
        self._router = _RVLSemanticRouter(
            name=self.name,
            routes=self._routes,
            redis_url=self.redis_url,
        )
        return self

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def match(self, query: str) -> Optional[str]:
        """
        Return the matched route name, or None if no route exceeds threshold.
        Does NOT invoke a handler.
        """
        if self._router is None:
            self.build()
        result = self._router(query)
        return result.name if result else None

    def route(self, query: str, **ctx) -> Tuple[Optional[str], Any]:
        """
        Match the query and invoke the corresponding handler.

        Returns (route_name, handler_return_value).
        If no route matches or no handler is registered, returns (None, None).
        """
        name = self.match(query)
        if name is None:
            print(f"[SemanticRouter] no match for {query!r}")
            return None, None

        handler = self._handlers.get(name)
        if handler is None:
            print(f"[SemanticRouter] matched route {name!r} (no handler)")
            return name, None

        print(f"[SemanticRouter] {query!r} → route={name!r}")
        return name, handler(query, **ctx)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def remove_route(self, name: str) -> None:
        self._routes = [r for r in self._routes if r.name != name]
        self._handlers.pop(name, None)
        self._router = None

    def list_routes(self) -> List[str]:
        return [r.name for r in self._routes]

    def __repr__(self) -> str:
        return (
            f"SemanticRouter(name={self.name!r}, "
            f"routes={self.list_routes()}, "
            f"threshold={self.distance_threshold})"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Define handlers ---

    def handle_tech_support(query: str, **_) -> str:
        return f"[Tech Support] I can help you troubleshoot: {query}"

    def handle_billing(query: str, **_) -> str:
        return f"[Billing] Let me pull up your account for: {query}"

    def handle_sales(query: str, **_) -> str:
        return f"[Sales] Great question about our products: {query}"

    def handle_small_talk(query: str, **_) -> str:
        return f"[Small Talk] {query} — Nice to chat!"

    # --- Build router ---
    router = SemanticRouter(name="customer_service", distance_threshold=0.35)

    router.add_route(
        "tech_support",
        ["my app is broken", "I can't login", "error message", "not working", "bug"],
        handle_tech_support,
    )
    router.add_route(
        "billing",
        ["invoice", "charge on my card", "refund", "subscription cost", "payment"],
        handle_billing,
    )
    router.add_route(
        "sales",
        ["pricing plans", "how much does it cost", "upgrade my plan", "features"],
        handle_sales,
    )
    router.add_route(
        "small_talk",
        ["hello", "how are you", "what's up", "good morning"],
        handle_small_talk,
    )

    print(router)
    print()

    test_queries = [
        "I keep getting a 500 error when I submit the form",
        "Can you send me a new invoice for last month?",
        "What does the Pro plan include?",
        "Hey! Good afternoon!",
        "I want to cancel my account",  # likely no strong match
    ]

    for q in test_queries:
        route_name, response = router.route(q)
        if response:
            print(f"  → {response}\n")
        else:
            print(f"  → (unmatched)\n")
