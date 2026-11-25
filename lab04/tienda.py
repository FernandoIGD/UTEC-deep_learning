"""Tienda simulada: precios, existencias, carrito, cotizaciones y pedidos con semilla fija."""

import numpy as np

SEMILLA = 42
PROPORCION_AGOTADOS = 0.08


class Tienda:
    """Estado de la tienda. Todas las validaciones de cantidades, existencias y presupuesto se hacen aquí.

    Args:
        ids: Identificadores de los productos del catálogo.
        semilla: Semilla para generar precios y existencias.
    """

    def __init__(self, ids, semilla=SEMILLA):
        self.ids = sorted(ids)
        self.semilla = semilla
        self.restaurar()

    def restaurar(self):
        """Vuelve al estado inicial generado con la semilla."""
        rng = np.random.default_rng(self.semilla)
        n = len(self.ids)
        precios = np.clip(np.round(np.exp(rng.normal(3.2, 0.9, n)), 2), 1.5, 900)
        existencias = rng.integers(1, 31, n)
        existencias[rng.random(n) < PROPORCION_AGOTADOS] = 0
        self.precios = dict(zip(self.ids, precios.tolist()))
        self.existencias = dict(zip(self.ids, existencias.tolist()))
        self.iniciales = dict(self.existencias)
        self.carrito = {}
        self.cotizaciones = {}
        self.pedidos = []

    def existe(self, product_id):
        return product_id in self.precios

    def disponibilidad(self, product_id):
        return {"precio": self.precios[product_id], "existencias": self.existencias[product_id]}

    def fijar_cantidad(self, product_id, cantidad):
        """Deja `cantidad` unidades del producto en el carrito; 0 lo elimina.

        Returns:
            Diccionario con `ok` y el detalle o el motivo del rechazo.
        """
        if not self.existe(product_id):
            return {"ok": False, "motivo": "El producto no existe en el catálogo."}
        if not isinstance(cantidad, int) or isinstance(cantidad, bool) or cantidad < 0:
            return {"ok": False, "motivo": "La cantidad debe ser un número entero mayor o igual a cero."}
        disponibles = self.existencias[product_id]
        if cantidad > disponibles:
            return {"ok": False, "motivo": f"Solo hay {disponibles} unidades disponibles.", "existencias": disponibles}
        if cantidad == 0:
            self.carrito.pop(product_id, None)
        else:
            self.carrito[product_id] = cantidad
        return {"ok": True, "cantidad": cantidad}

    def total_carrito(self):
        return round(sum(self.precios[pid] * cantidad for pid, cantidad in self.carrito.items()), 2)

    def problemas(self, items, presupuesto):
        encontrados = []
        for pid, cantidad, precio in items:
            if self.existencias[pid] < cantidad:
                encontrados.append({"product_id": pid, "problema": "existencias", "pedidas": cantidad, "disponibles": self.existencias[pid]})
            if self.precios[pid] != precio:
                encontrados.append({"product_id": pid, "problema": "precio", "cotizado": precio, "actual": self.precios[pid]})
        total = round(sum(cantidad * precio for _, cantidad, precio in items), 2)
        if presupuesto is not None and total > presupuesto:
            encontrados.append({"problema": "presupuesto", "total": total, "presupuesto": presupuesto})
        return encontrados

    def cotizar(self, presupuesto=None):
        """Crea una cotización del carrito actual con los precios vigentes.

        Args:
            presupuesto: Monto máximo indicado por el usuario, o None.

        Returns:
            Diccionario con el identificador, los ítems, el total y los problemas encontrados.
        """
        if not self.carrito:
            return {"ok": False, "motivo": "El carrito está vacío."}
        items = [(pid, cantidad, self.precios[pid]) for pid, cantidad in sorted(self.carrito.items())]
        total = round(sum(cantidad * precio for _, cantidad, precio in items), 2)
        cotizacion_id = f"C{len(self.cotizaciones) + 1}"
        problemas = self.problemas(items, presupuesto)
        for anterior in self.cotizaciones.values():
            if anterior["estado"] == "pendiente":
                anterior["estado"] = "reemplazada"
        self.cotizaciones[cotizacion_id] = {"items": items, "total": total, "estado": "pendiente" if not problemas else "con_problemas"}
        return {"ok": not problemas, "cotizacion_id": cotizacion_id, "total": total, "items": items, "problemas": problemas}

    def confirmar(self, cotizacion_id, presupuesto=None):
        """Registra el pedido de una cotización después de volver a validar todas sus condiciones.

        El pedido se registra completo o no se registra. Confirmar dos veces la misma cotización
        devuelve el pedido existente y no crea otro.

        Args:
            cotizacion_id: Identificador devuelto por `cotizar`.
            presupuesto: Monto máximo indicado por el usuario, o None.

        Returns:
            Diccionario con `ok` y el pedido, o con los problemas que impiden registrarlo.
        """
        cotizacion = self.cotizaciones.get(cotizacion_id)
        if cotizacion is None:
            return {"ok": False, "motivo": "La cotización no existe."}
        if cotizacion["estado"] == "confirmada":
            pedido = next(p for p in self.pedidos if p["cotizacion_id"] == cotizacion_id)
            return {"ok": True, "repetido": True, "pedido": pedido}
        if cotizacion["estado"] != "pendiente":
            return {"ok": False, "motivo": "La cotización tenía problemas o fue reemplazada. Se debe cotizar de nuevo."}
        actual = sorted(self.carrito.items())
        if actual != [(pid, cantidad) for pid, cantidad, _ in cotizacion["items"]]:
            return {"ok": False, "motivo": "El carrito cambió después de la cotización. Se debe cotizar de nuevo."}
        problemas = self.problemas(cotizacion["items"], presupuesto)
        if problemas:
            cotizacion["estado"] = "invalida"
            return {"ok": False, "motivo": "Las condiciones cambiaron desde la cotización.", "problemas": problemas}

        for pid, cantidad, _ in cotizacion["items"]:
            self.existencias[pid] -= cantidad
        pedido = {"pedido_id": f"P{len(self.pedidos) + 1}", "cotizacion_id": cotizacion_id, "items": cotizacion["items"], "total": cotizacion["total"]}
        self.pedidos.append(pedido)
        cotizacion["estado"] = "confirmada"
        self.carrito = {}
        return {"ok": True, "repetido": False, "pedido": pedido}

    def cambios(self):
        """Estado final resumido: existencias modificadas, carrito, cotizaciones y pedidos."""
        existencias = {pid: [self.iniciales[pid], n] for pid, n in self.existencias.items() if n != self.iniciales[pid]}
        return {"existencias": existencias, "carrito": dict(self.carrito), "cotizaciones": self.cotizaciones, "pedidos": self.pedidos}
